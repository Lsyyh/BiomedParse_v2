#!/usr/bin/env python3
"""
Direct model diagnostic test - bypasses the FastAPI service.

Tests the BiomedParse v2 model directly on demo fundus images to diagnose
why the segmentation service produces poor results.

Usage:
    CUDA_VISIBLE_DEVICES=1 python direct_model_test.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Add project root to path
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


def main():
    print("=" * 60)
    print("BiomedParse v2 Direct Model Diagnostic Test")
    print("=" * 60)
    print()

    # Setup paths
    demo_dir = Path(PROJECT_ROOT) / "biomedparse_datasets" / "BiomedParseData-Demo"
    fundus_image_path = demo_dir / "demo" / "23_fundus_retinal.png"
    gt_masks = {
        "optic disc": demo_dir / "demo_mask" / "23_fundus_retinal_optic+disc.png",
        "optic cup": demo_dir / "demo_mask" / "23_fundus_retinal_optic+cup.png",
    }

    if not fundus_image_path.exists():
        print(f"ERROR: Demo image not found: {fundus_image_path}")
        sys.exit(1)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print()

    # Load model (same as service)
    print("1. Loading model...")
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="diagnostic_test")
    cfg = compose(config_name="biomedparse")
    model = hydra.utils.instantiate(cfg, _convert_="object")

    # Load v2 weights
    print("   Downloading v2 weights...")
    checkpoint_path = hf_hub_download(
        repo_id="microsoft/BiomedParse",
        filename="biomedparse_v2.ckpt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint
    state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"   Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    print("   Model loaded!")
    print()

    # Load image
    print("2. Loading demo image...")
    image_pil = Image.open(fundus_image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    print(f"   Original size: {orig_w}x{orig_h}")

    # Resize to 1024x1024 (same as service)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)
    ])
    image_resize = transform(image_pil)
    image_array = np.asarray(image_resize)
    image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"   Tensor shape: {image_tensor.shape}")
    print(f"   Tensor range: [{image_tensor.min():.1f}, {image_tensor.max():.1f}]")
    print(f"   Tensor mean: {image_tensor.float().mean():.2f}")
    print()

    # Test different text prompts
    prompts = ["optic disc", "optic cup"]

    for prompt in prompts:
        print(f"3. Testing prompt: '{prompt}'")
        gt_path = gt_masks.get(prompt)

        # Run inference (same as service)
        inputs = {"image": image_tensor, "text": [prompt]}
        with torch.no_grad():
            results = model(inputs, mode="eval")

        pred_gmasks = results["predictions"]["pred_gmasks"]
        print(f"   pred_gmasks shape: {pred_gmasks.shape}")
        print(f"   pred_gmasks range: [{pred_gmasks.min():.4f}, {pred_gmasks.max():.4f}]")
        print(f"   pred_gmasks mean: {pred_gmasks.mean():.6f}")

        # Interpolate to original size
        pred_mask = F.interpolate(
            pred_gmasks,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        pred_mask_prob = pred_mask.sigmoid().cpu().numpy()[0, 0]

        print(f"   After sigmoid range: [{pred_mask_prob.min():.6f}, {pred_mask_prob.max():.6f}]")
        print(f"   After sigmoid mean: {pred_mask_prob.mean():.6f}")
        print(f"   Pixels > 0.5: {(pred_mask_prob > 0.5).sum()}")
        print(f"   Pixels > 0.1: {(pred_mask_prob > 0.1).sum()}")
        print(f"   Pixels > 0.01: {(pred_mask_prob > 0.01).sum()}")

        # Save probability mask
        prob_output = f"diag_prob_{prompt.replace(' ', '_')}.png"
        prob_vis = (pred_mask_prob * 255).astype(np.uint8)
        Image.fromarray(prob_vis, mode="L").save(prob_output)
        print(f"   Saved probability mask: {prob_output}")

        # Save binary mask
        binary_mask = (pred_mask_prob > 0.5).astype(np.uint8) * 255
        binary_output = f"diag_binary_{prompt.replace(' ', '_')}.png"
        Image.fromarray(binary_mask, mode="L").save(binary_output)
        print(f"   Saved binary mask: {binary_output}")

        # Compare with ground truth
        if gt_path and gt_path.exists():
            gt_mask = np.array(Image.open(gt_path).convert("L"))
            gt_binary = (gt_mask > 127).astype(np.uint8)
            pred_binary = (pred_mask_prob > 0.5).astype(np.uint8)
            iou = compute_iou(pred_binary, gt_binary)
            dice = compute_dice(pred_binary, gt_binary)
            print(f"   IoU vs GT: {iou:.4f}")
            print(f"   Dice vs GT: {dice:.4f}")

            # Also try lower thresholds
            for thresh in [0.1, 0.2, 0.3, 0.4]:
                pred_t = (pred_mask_prob > thresh).astype(np.uint8)
                iou_t = compute_iou(pred_t, gt_binary)
                dice_t = compute_dice(pred_t, gt_binary)
                print(f"   Threshold {thresh}: IoU={iou_t:.4f}, Dice={dice_t:.4f}")

        print()

    # Also test with lower threshold (0.3) to see if mask is there but weak
    print("4. Testing with lowered threshold (0.3)...")
    for prompt in prompts:
        gt_path = gt_masks.get(prompt)
        if not gt_path or not gt_path.exists():
            continue

        gt_mask = np.array(Image.open(gt_path).convert("L"))
        gt_binary = (gt_mask > 127).astype(np.uint8)

        inputs = {"image": image_tensor, "text": [prompt]}
        with torch.no_grad():
            results = model(inputs, mode="eval")
        pred_gmasks = results["predictions"]["pred_gmasks"]
        pred_mask = F.interpolate(pred_gmasks, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        pred_mask_prob = pred_mask.sigmoid().cpu().numpy()[0, 0]

        for thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_binary = (pred_mask_prob > thresh).astype(np.uint8)
            iou = compute_iou(pred_binary, gt_binary)
            dice = compute_dice(pred_binary, gt_binary)
            n_pixels = pred_binary.sum()
            print(f"   '{prompt}' thresh={thresh}: IoU={iou:.4f}, Dice={dice:.4f}, pixels={n_pixels}")

    print()
    print("=" * 60)
    print("Diagnostic test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
