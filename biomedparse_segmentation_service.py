#!/usr/bin/env python3
"""
BiomedParse Segmentation Service (v1 + v2)

FastAPI service for biomedical image segmentation using BiomedParse models.
- 2D images: v1 model with vl_similarity query matching
- 3D volumes: v2 model with object_existence gating

Exposes REST API for MedAgent-Ultra integration.
"""

import base64
import gc
import io
import os
import sys
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    from huggingface_hub import hf_hub_download
    from utils import slice_nms, process_input, process_output
    from src.model.biomedparse import process_multi_prompts, tile_feature
    BIOMEDPARSE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    BIOMEDPARSE_AVAILABLE = False


# Pydantic models for API
class SegmentationRequest(BaseModel):
    """Request model for segmentation endpoint."""
    image: str  # base64 encoded image
    text_prompt: str  # text prompt for segmentation
    dimension: str = "2d"  # "2d" for fundus/pathology images, "3d" for CT/MRI volumes


class SegmentationResponse(BaseModel):
    """Response model for segmentation endpoint."""
    mask: Optional[str] = None  # base64 encoded grayscale PNG mask
    success: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str


class VolumeSegmentationRequest(BaseModel):
    """Request model for 3D volume segmentation endpoint."""
    volume: str  # base64 encoded .npz file containing 'imgs' and 'text_prompts'


class VolumeSegmentationResponse(BaseModel):
    """Response model for 3D volume segmentation endpoint."""
    segmentation: Optional[str] = None  # base64 encoded .npz file with 'segs'
    shape: Optional[list] = None  # shape of the segmentation volume
    success: bool
    error: Optional[str] = None


# Global model variables
model_2d = None  # v1 model for 2D
model_3d = None  # v2 model for 3D
device = None


def convert_v1_to_v2_keys(v1_sd, v2_keys):
    """Convert v1 checkpoint keys to v2 model key format."""
    mapping = {}
    for v1_key in v1_sd.keys():
        v2_key = v1_key
        v2_key = v2_key.replace('lang_encoder.lang_encoder', 'language_encoder.encoder_transformer')
        v2_key = v2_key.replace('lang_encoder', 'language_encoder')
        v2_key = v2_key.replace('query_feat.weight', 'query_feat_.weight')
        v2_key = v2_key.replace('query_embed.weight', 'query_embed_.weight')
        if 'biomed_encoder' in v2_key:
            continue
        if v2_key in v2_keys:
            mapping[v1_key] = v2_key
    return mapping


def vl_similarity(image_feat, text_feat, temperature=1):
    """Compute vision-language similarity (from v1 codebase)."""
    logits = torch.matmul(image_feat, text_feat.t())
    logits = temperature.exp().clamp(max=100) * logits
    return logits


def load_model():
    """Load both v1 (2D) and v2 (3D) models on startup."""
    global model_2d, model_3d, device

    if not BIOMEDPARSE_AVAILABLE:
        logger.error("Required modules not available. Model not loaded.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.info("Using CPU")

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="segmentation_service")

    # Load v1 model for 2D
    try:
        logger.info("Loading BiomedParse v1 2D model...")
        cfg_2d = compose(config_name="biomedparse_v1_2d")
        model_2d = hydra.utils.instantiate(cfg_2d, _convert_="object")
        v2_keys = set(model_2d.state_dict().keys())

        v1_path = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v1.pt")
        v1_ckpt = torch.load(v1_path, map_location="cpu")
        v1_sd = v1_ckpt if not isinstance(v1_ckpt, dict) else v1_ckpt.get("state_dict", v1_ckpt.get("model", v1_ckpt))

        mapping = convert_v1_to_v2_keys(v1_sd, v2_keys)
        mapped_sd = {mapping[k]: v1_sd[k] for k in mapping}
        missing, _ = model_2d.load_state_dict(mapped_sd, strict=False)
        model_2d = model_2d.to(device).eval()
        logger.info(f"v1 2D model loaded ({len(mapping)} keys mapped, {len(missing)} missing)")
    except Exception as e:
        logger.error(f"Failed to load v1 2D model: {e}")
        import traceback
        traceback.print_exc()

    # Load v2 model for 3D
    try:
        logger.info("Loading BiomedParse v2 3D model...")
        cfg_3d = compose(config_name="biomedparse_3D")
        model_3d = hydra.utils.instantiate(cfg_3d, _convert_="object")

        v2_path = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v2.ckpt")
        v2_ckpt = torch.load(v2_path, map_location=device)

        if isinstance(v2_ckpt, dict):
            state_dict = v2_ckpt.get("state_dict", v2_ckpt.get("model", v2_ckpt))
        else:
            state_dict = v2_ckpt

        state_dict = {
            k[6:] if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }

        missing, unexpected = model_3d.load_state_dict(state_dict, strict=False)
        model_3d = model_3d.to(device).eval()
        logger.info(f"v2 3D model loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
    except Exception as e:
        logger.error(f"Failed to load v2 3D model: {e}")
        import traceback
        traceback.print_exc()


def decode_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")


def encode_mask(mask_array: np.ndarray) -> str:
    """Encode numpy mask to base64 string."""
    try:
        mask_image = Image.fromarray(mask_array, mode="L")
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_string
    except Exception as e:
        raise ValueError(f"Failed to encode mask: {e}")


@torch.no_grad()
def run_inference_2d(image: Image.Image, text_prompt: str) -> np.ndarray:
    """
    Run v1 model inference on 2D image with vl_similarity query matching.

    Args:
        image: PIL Image (RGB)
        text_prompt: text prompt for segmentation

    Returns:
        numpy array of binary mask (H, W), values 0 or 255
    """
    global model_2d, device

    if model_2d is None:
        raise RuntimeError("v1 2D model not loaded")

    orig_w, orig_h = image.size

    # Resize to 1024x1024
    image_resized = image.resize((1024, 1024), Image.BICUBIC)
    image_array = np.asarray(image_resized)
    image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalize
    image_tensor = (image_tensor - model_2d.pixel_mean) / model_2d.pixel_std

    # Get image features
    image_embedding = model_2d.backbone(image_tensor)

    # Get text embeddings
    model_2d.sem_seg_head.predictor.language_encoder.get_text_embeddings(
        model_2d.sem_seg_head.classes, is_eval=True
    )

    text, num_prompts = process_multi_prompts([text_prompt])

    gtext = model_2d.sem_seg_head.predictor.language_encoder.get_text_token_embeddings(
        text, name="grounding", token=False, norm=False
    )
    token_emb = gtext["token_emb"]
    tokens = gtext["tokens"]
    class_emb = gtext["class_emb"]
    query_emb = torch.nn.utils.rnn.pad_sequence(
        [
            _token_emb[_tokens.bool()]
            for _token_emb, _tokens in zip(token_emb, tokens["attention_mask"])
        ],
        padding_value=-1,
    )
    non_zero_query_mask = query_emb.sum(dim=-1) == -query_emb.shape[-1]
    query_emb[non_zero_query_mask] = 0

    extra = {
        "grounding_tokens": query_emb,
        "grounding_nonzero_mask": non_zero_query_mask.t(),
    }

    # Get mask features
    mask_features, _, multi_scale_features = (
        model_2d.sem_seg_head.pixel_decoder.forward_features(image_embedding)
    )

    # Tile features for number of prompts
    P = int(num_prompts[0])
    mask_features = tile_feature(mask_features, P)
    multi_scale_features = [tile_feature(f, P) for f in multi_scale_features]

    # Run predictor
    predictions = model_2d.sem_seg_head.predictor(
        x=multi_scale_features, mask_features=mask_features, mask=None, extra=extra
    )

    # Get per-query masks and visual embeddings
    pred_gmasks = predictions["pred_gmasks"]    # (P, num_queries, H, W)
    pred_captions = predictions["pred_captions"]  # (P, num_queries, dim)

    # vl_similarity matching
    logit_scale = model_2d.sem_seg_head.predictor.language_encoder.logit_scale

    v_emb = pred_captions[0]  # (num_queries, dim)
    t_emb = class_emb         # (1, dim)

    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)

    out_prob = vl_similarity(v_emb, t_emb, temperature=logit_scale)
    matched_id = out_prob.max(0)[1].item()

    best_mask = pred_gmasks[0, matched_id]  # (H, W)

    # Interpolate to original size
    pred_mask = F.interpolate(
        best_mask.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False
    )

    pred_mask_prob = pred_mask.cpu().numpy()[0, 0]
    binary_mask = (pred_mask_prob > 0.5).astype(np.uint8) * 255

    return binary_mask


@torch.no_grad()
def run_inference_3d(image: Image.Image, text_prompt: str) -> np.ndarray:
    """
    Run v2 model inference on 2D image using 3D model path.

    Args:
        image: PIL Image (RGB)
        text_prompt: text prompt for segmentation

    Returns:
        numpy array of binary mask (H, W), values 0 or 255
    """
    global model_3d, device

    if model_3d is None:
        raise RuntimeError("v2 3D model not loaded")

    width, height = image.size

    # Convert to grayscale
    image_gray = image.convert("L")
    image_resized = image_gray.resize((512, 512), Image.BICUBIC)
    image_array = np.asarray(image_resized, dtype=np.uint8)

    # Create single-slice volume: (batch=1, D=1, H=512, W=512)
    image_tensor = torch.from_numpy(image_array.copy()).unsqueeze(0).unsqueeze(0).to(device)
    image_tensor = image_tensor.int()

    inputs = {
        "image": image_tensor,
        "text": [text_prompt],
    }

    results = model_3d(inputs, mode="eval", slice_batch_size=1)

    pred_gmasks = results["predictions"]["pred_gmasks"]
    object_existence = results["predictions"]["object_existence"]

    mask_probs = pred_gmasks.sigmoid()
    obj_exist = (object_existence.sigmoid() > 0.5).int()
    obj_exist = obj_exist.unsqueeze(-1).unsqueeze(-1)
    mask_probs = mask_probs * obj_exist

    mask_probs = mask_probs[0, 0]

    pred_mask = F.interpolate(
        mask_probs.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )

    pred_mask_prob = pred_mask.cpu().numpy()[0, 0]
    binary_mask = (pred_mask_prob > 0.5).astype(np.uint8) * 255

    return binary_mask


def merge_multiclass_masks(masks, ids):
    """Merge multi-class masks into a single label map."""
    bg_mask = 0.5 * torch.ones_like(masks[0:1])
    keep_masks = torch.cat([bg_mask, masks], dim=0)
    class_mask = keep_masks.argmax(dim=0)
    id_map = {j + 1: int(ids[j]) for j in range(len(ids)) if j + 1 != int(ids[j])}
    if len(id_map) > 0:
        orig_mask = class_mask.clone()
        for j in id_map:
            class_mask[orig_mask == j] = id_map[j]
    return class_mask


def postprocess_3d(model_outputs, object_existence, threshold=0.5, do_nms=True):
    """Postprocess 3D model outputs with sigmoid, gating, and NMS."""
    if do_nms and model_outputs.shape[0] > 1:
        return slice_nms(
            model_outputs.sigmoid(),
            object_existence.sigmoid(),
            iou_threshold=0.5,
            score_threshold=threshold,
        )
    mask = (model_outputs.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)
    return mask


@torch.no_grad()
def run_inference_3d_volume(volume_npz_bytes: bytes) -> tuple:
    """
    Run full 3D volume inference using v2 model.

    Args:
        volume_npz_bytes: raw bytes of .npz file

    Returns:
        (segmentation_npz_bytes, shape_list) - compressed npz bytes and output shape
    """
    global model_3d, device

    if model_3d is None:
        raise RuntimeError("v2 3D model not loaded")

    # Load npz data
    npz_data = np.load(io.BytesIO(volume_npz_bytes), allow_pickle=True)
    imgs = npz_data["imgs"]
    text_prompts = npz_data["text_prompts"].item()

    ids = [int(_) for _ in text_prompts.keys() if _ != "instance_label"]
    ids.sort()
    text = "[SEP]".join([text_prompts[str(i)] for i in ids])

    logger.info(f"3D volume shape: {imgs.shape}, classes: {len(ids)}")

    # Process input: pad and resize to 512
    imgs_padded, pad_width, padded_size, valid_axis = process_input(imgs, 512)
    imgs_tensor = imgs_padded.to(device).int()

    input_tensor = {
        "image": imgs_tensor.unsqueeze(0),
        "text": [text],
    }

    # Run model inference
    output = model_3d(input_tensor, mode="eval", slice_batch_size=4)

    mask_preds = output["predictions"]["pred_gmasks"]
    mask_preds = F.interpolate(
        mask_preds,
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    mask_preds = postprocess_3d(mask_preds, output["predictions"]["object_existence"])
    mask_preds = merge_multiclass_masks(mask_preds, ids)
    segs = process_output(mask_preds, pad_width, padded_size, valid_axis)

    # Cleanup GPU memory
    del imgs_tensor, input_tensor, output, mask_preds
    gc.collect()
    torch.cuda.empty_cache()

    # Compress result to npz bytes
    buf = io.BytesIO()
    np.savez_compressed(buf, segs=segs)
    buf.seek(0)

    return buf.read(), list(segs.shape)


# Create FastAPI app
app = FastAPI(
    title="BiomedParse Segmentation Service",
    description="REST API for biomedical image segmentation using BiomedParse (v1 for 2D, v2 for 3D)",
    version="3.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_model()


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(request: SegmentationRequest):
    """
    Segment biomedical image using text prompt.

    Args:
        request: SegmentationRequest with base64 image, text prompt, and dimension

    Returns:
        SegmentationResponse with base64 mask or error
    """
    try:
        logger.info(f"Segmentation request: prompt='{request.text_prompt}', dimension={request.dimension}")
        image = decode_image(request.image)

        if request.dimension == "2d":
            binary_mask = run_inference_2d(image, request.text_prompt)
        elif request.dimension == "3d":
            binary_mask = run_inference_3d(image, request.text_prompt)
        else:
            raise ValueError(f"Invalid dimension: {request.dimension}. Must be '2d' or '3d'.")

        mask_base64 = encode_mask(binary_mask)
        logger.info("Segmentation completed successfully")

        return SegmentationResponse(
            mask=mask_base64,
            success=True,
            error=None
        )

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return SegmentationResponse(
            mask=None,
            success=False,
            error=str(e)
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status_2d = "loaded" if model_2d is not None else "not loaded"
    status_3d = "loaded" if model_3d is not None else "not loaded"
    return HealthResponse(status=f"2D(v1): {status_2d}, 3D(v2): {status_3d}")


@app.post("/segment_volume", response_model=VolumeSegmentationResponse)
async def segment_volume(request: VolumeSegmentationRequest):
    """
    Segment 3D volume using v2 model.

    Args:
        request: VolumeSegmentationRequest with base64 encoded .npz file

    Returns:
        VolumeSegmentationResponse with base64 encoded .npz segmentation
    """
    try:
        logger.info("Processing 3D volume segmentation request")

        # Decode base64 npz
        volume_bytes = base64.b64decode(request.volume)

        # Run 3D inference
        seg_bytes, seg_shape = run_inference_3d_volume(volume_bytes)

        seg_base64 = base64.b64encode(seg_bytes).decode("utf-8")
        logger.info(f"3D segmentation completed, shape: {seg_shape}")

        return VolumeSegmentationResponse(
            segmentation=seg_base64,
            shape=seg_shape,
            success=True,
            error=None,
        )

    except Exception as e:
        logger.error(f"3D segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return VolumeSegmentationResponse(
            segmentation=None,
            shape=None,
            success=False,
            error=str(e),
        )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "BiomedParse Segmentation Service",
        "version": "3.0.0",
        "models": {
            "2d": "BiomedParse v1 (vl_similarity query matching)",
            "3d": "BiomedParse v2 (object_existence gating)"
        },
        "endpoints": {
            "POST /segment": "Segment 2D image with text prompt (dimension='2d' or '3d')",
            "POST /segment_volume": "Segment 3D volume (.npz) with v2 model",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "biomedparse_segmentation_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
