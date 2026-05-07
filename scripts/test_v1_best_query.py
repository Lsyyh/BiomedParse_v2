#!/usr/bin/env python3
"""
Test v1 model with best-query selection instead of averaging all queries.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return intersection / union if union > 0 else 0.0


def compute_dice(mask1, mask2):
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    return 2 * intersection / total if total > 0 else 0.0


def convert_v1_to_v2_keys(v1_sd, v2_keys):
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


@torch.no_grad()
def run_inference_vl_match(model, image_tensor, text_prompt, device):
    """
    Run inference using v1's vl_similarity matching mechanism.
    Selects the query whose visual embedding best matches the text prompt.
    """
    image = image_tensor

    # Normalize
    image = (image - model.pixel_mean) / model.pixel_std

    # Get image features
    image_embedding = model.backbone(image)

    # Get text embeddings
    model.sem_seg_head.predictor.language_encoder.get_text_embeddings(
        model.sem_seg_head.classes, is_eval=True
    )

    # Process text prompt
    from src.model.biomedparse import process_multi_prompts, tile_feature
    text, num_prompts = process_multi_prompts([text_prompt])

    gtext = model.sem_seg_head.predictor.language_encoder.get_text_token_embeddings(
        text, name="grounding", token=False, norm=False
    )
    token_emb = gtext["token_emb"]
    tokens = gtext["tokens"]
    class_emb = gtext["class_emb"]  # text CLS embedding
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
        model.sem_seg_head.pixel_decoder.forward_features(image_embedding)
    )

    # Tile features for number of prompts
    P = int(num_prompts[0])
    mask_features = tile_feature(mask_features, P)
    multi_scale_features = [tile_feature(f, P) for f in multi_scale_features]

    # Run predictor
    predictions = model.sem_seg_head.predictor(
        x=multi_scale_features, mask_features=mask_features, mask=None, extra=extra
    )

    # Get per-query masks and visual embeddings
    pred_gmasks = predictions["pred_gmasks"]    # (P, num_queries, H, W)
    pred_captions = predictions["pred_captions"]  # (P, num_queries, dim)

    # vl_similarity matching (from v1 inference)
    logit_scale = model.sem_seg_head.predictor.language_encoder.logit_scale

    best_masks = []
    for p in range(pred_gmasks.shape[0]):
        v_emb = pred_captions[p]  # (num_queries, dim)
        t_emb = class_emb         # (1, dim) or (num_texts, dim)

        # Normalize
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)

        # Compute similarity and find best matching query
        out_prob = vl_similarity(v_emb, t_emb, temperature=logit_scale)
        matched_id = out_prob.max(0)[1].item()  # scalar index of best matching query

        best_mask = pred_gmasks[p, matched_id]  # (H, W)
        best_masks.append(best_mask)

    best_masks = torch.stack(best_masks, dim=0).unsqueeze(1)  # (num_prompts, 1, H, W)
    return best_masks


def main():
    print("=" * 60)
    print("Test v1 Model with Best-Query Selection")
    print("=" * 60)
    print()

    demo_dir = Path(PROJECT_ROOT) / "biomedparse_datasets" / "BiomedParseData-Demo"
    fundus_image_path = demo_dir / "demo" / "23_fundus_retinal.png"
    gt_masks = {
        "optic disc": demo_dir / "demo_mask" / "23_fundus_retinal_optic+disc.png",
        "optic cup": demo_dir / "demo_mask" / "23_fundus_retinal_optic+cup.png",
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print()

    # Load v1-compatible 2D model
    print("1. Loading v1-compatible 2D model...")
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="test_v1_bestquery")
    cfg = compose(config_name="biomedparse_v1_2d")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    v2_keys = set(model.state_dict().keys())

    v1_path = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v1.pt")
    v1_ckpt = torch.load(v1_path, map_location="cpu")
    v1_sd = v1_ckpt if not isinstance(v1_ckpt, dict) else v1_ckpt.get("state_dict", v1_ckpt.get("model", v1_ckpt))

    mapping = convert_v1_to_v2_keys(v1_sd, v2_keys)
    mapped_sd = {mapping[k]: v1_sd[k] for k in mapping}
    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    print(f"   Mapped {len(mapping)} keys, Missing: {len(missing)}")
    model = model.to(device).eval()
    print("   Model loaded!")
    print()

    # Load image
    print("2. Loading demo image...")
    image_pil = Image.open(fundus_image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    print(f"   Original size: {orig_w}x{orig_h}")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)
    ])
    image_resize = transform(image_pil)
    image_array = np.asarray(image_resize)
    image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    print()

    # Test prompts
    prompts = ["optic disc", "optic cup"]

    for prompt in prompts:
        print(f"3. Testing prompt: '{prompt}' (vl_similarity matching)")
        gt_path = gt_masks.get(prompt)

        pred_gmasks = run_inference_vl_match(model, image_tensor, prompt, device)

        print(f"   pred_gmasks shape: {pred_gmasks.shape}")
        print(f"   pred_gmasks range: [{pred_gmasks.min():.4f}, {pred_gmasks.max():.4f}]")

        # Interpolate to original size
        pred_mask = F.interpolate(
            pred_gmasks, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )
        pred_mask_prob = pred_mask.sigmoid().cpu().numpy()[0, 0]

        print(f"   After sigmoid range: [{pred_mask_prob.min():.6f}, {pred_mask_prob.max():.6f}]")
        print(f"   Pixels > 0.5: {(pred_mask_prob > 0.5).sum()}")

        # Save
        prob_vis = (pred_mask_prob * 255).astype(np.uint8)
        Image.fromarray(prob_vis, mode="L").save(f"test_bestq_{prompt.replace(' ', '_')}_prob.png")
        binary_mask = (pred_mask_prob > 0.5).astype(np.uint8) * 255
        Image.fromarray(binary_mask, mode="L").save(f"test_bestq_{prompt.replace(' ', '_')}_binary.png")

        # Compare with GT
        if gt_path and gt_path.exists():
            gt_mask = np.array(Image.open(gt_path).convert("L"))
            gt_binary = (gt_mask > 127).astype(np.uint8)
            for thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
                pred_t = (pred_mask_prob > thresh).astype(np.uint8)
                iou = compute_iou(pred_t, gt_binary)
                dice = compute_dice(pred_t, gt_binary)
                print(f"   Threshold {thresh}: IoU={iou:.4f}, Dice={dice:.4f}")

        print()

    print("=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
