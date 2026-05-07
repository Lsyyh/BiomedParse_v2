#!/usr/bin/env python3
"""
Test 2D model with v1 weights (with key mapping) for fundus segmentation.
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
    """Convert v1 checkpoint keys to v2 model key format."""
    mapping = {}
    for v1_key in v1_sd.keys():
        v2_key = v1_key
        # lang_encoder -> language_encoder
        v2_key = v2_key.replace('lang_encoder', 'language_encoder')
        # language_encoder.lang_encoder -> language_encoder.encoder_transformer
        v2_key = v2_key.replace('language_encoder.lang_encoder', 'language_encoder.encoder_transformer')
        # query_feat -> query_feat_ (trailing underscore)
        v2_key = v2_key.replace('query_feat.weight', 'query_feat_.weight')
        v2_key = v2_key.replace('query_embed.weight', 'query_embed_.weight')
        # Skip biomed_encoder keys (v1-specific)
        if 'biomed_encoder' in v2_key:
            continue
        if v2_key in v2_keys:
            mapping[v1_key] = v2_key
    return mapping


def main():
    print("=" * 60)
    print("Test 2D Model with v1 Weights (Mapped)")
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

    # Load 2D model
    print("1. Loading 2D model...")
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="test_v1_mapped")
    cfg = compose(config_name="biomedparse")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    v2_keys = set(model.state_dict().keys())

    # Load v1 weights with key mapping
    print("   Downloading v1 weights...")
    v1_path = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v1.pt")
    v1_ckpt = torch.load(v1_path, map_location="cpu")
    v1_sd = v1_ckpt if not isinstance(v1_ckpt, dict) else v1_ckpt.get("state_dict", v1_ckpt.get("model", v1_ckpt))

    # Convert keys
    mapping = convert_v1_to_v2_keys(v1_sd, v2_keys)
    mapped_sd = {mapping[k]: v1_sd[k] for k in mapping}

    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    print(f"   Mapped {len(mapping)} v1 keys to v2 format")
    print(f"   Missing keys: {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
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
    print(f"   Tensor shape: {image_tensor.shape}")
    print()

    # Test prompts
    prompts = ["optic disc", "optic cup"]

    for prompt in prompts:
        print(f"3. Testing prompt: '{prompt}'")
        gt_path = gt_masks.get(prompt)

        inputs = {"image": image_tensor, "text": [prompt]}
        with torch.no_grad():
            results = model(inputs, mode="eval")

        pred_gmasks = results["predictions"]["pred_gmasks"]
        print(f"   pred_gmasks shape: {pred_gmasks.shape}")
        print(f"   pred_gmasks range: [{pred_gmasks.min():.4f}, {pred_gmasks.max():.4f}]")

        # Interpolate to original size
        pred_mask = F.interpolate(
            pred_gmasks, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )
        pred_mask_prob = pred_mask.sigmoid().cpu().numpy()[0, 0]

        print(f"   After sigmoid range: [{pred_mask_prob.min():.6f}, {pred_mask_prob.max():.6f}]")
        print(f"   Pixels > 0.5: {(pred_mask_prob > 0.5).sum()}")
        print(f"   Pixels > 0.1: {(pred_mask_prob > 0.1).sum()}")

        # Save
        prob_vis = (pred_mask_prob * 255).astype(np.uint8)
        Image.fromarray(prob_vis, mode="L").save(f"test_v1mapped_{prompt.replace(' ', '_')}_prob.png")
        binary_mask = (pred_mask_prob > 0.5).astype(np.uint8) * 255
        Image.fromarray(binary_mask, mode="L").save(f"test_v1mapped_{prompt.replace(' ', '_')}_binary.png")

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
