#!/usr/bin/env python3
"""
Test using 3D model to segment 2D fundus images.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

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
    print("Test: 3D Model for 2D Fundus Segmentation")
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

    # Load 3D model
    print("1. Loading 3D model...")
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="test_3d_2d")
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

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
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print("   Model loaded!")
    print()

    # Load 2D fundus image
    print("2. Processing 2D fundus image...")
    image_pil = Image.open(fundus_image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    print(f"   Original size: {orig_w}x{orig_h}")

    # Convert to grayscale
    image_gray = image_pil.convert("L")
    print(f"   Converted to grayscale")

    # Resize to 512x512
    image_resized = image_gray.resize((512, 512), Image.BICUBIC)
    image_array = np.asarray(image_resized, dtype=np.uint8)
    print(f"   Resized to 512x512, range: [{image_array.min()}, {image_array.max()}]")

    # Create single-slice volume: (batch=1, D=1, H=512, W=512)
    # 3D model expects (batch, D, H, W) format, not (batch, channel, D, H, W)
    image_tensor = torch.from_numpy(image_array.copy()).unsqueeze(0).unsqueeze(0).to(device)
    image_tensor = image_tensor.int()
    print(f"   Tensor shape: {image_tensor.shape}")
    print()

    # Test prompts
    prompts = ["optic disc", "optic cup"]

    for prompt in prompts:
        print(f"3. Testing prompt: '{prompt}'")
        gt_path = gt_masks.get(prompt)

        inputs = {"image": image_tensor, "text": [prompt]}

        with torch.no_grad():
            results = model(inputs, mode="eval", slice_batch_size=1)

        pred_gmasks = results["predictions"]["pred_gmasks"]
        object_existence = results["predictions"]["object_existence"]

        print(f"   pred_gmasks shape: {pred_gmasks.shape}")
        print(f"   pred_gmasks range: [{pred_gmasks.min():.4f}, {pred_gmasks.max():.4f}]")
        print(f"   object_existence shape: {object_existence.shape}")
        print(f"   object_existence range: [{object_existence.min():.4f}, {object_existence.max():.4f}]")
        print(f"   object_existence sigmoid: {object_existence.sigmoid()}")

        # Postprocess - try both with and without object_existence gating
        mask_probs_raw = pred_gmasks.sigmoid()

        # Without object_existence gating
        mask_probs = mask_probs_raw

        # Print object_existence info
        obj_exist_prob = object_existence.sigmoid()
        print(f"   object_existence prob: {obj_exist_prob}")
        print(f"   Without gating, mask range: [{mask_probs.min():.4f}, {mask_probs.max():.4f}]")

        # Take single slice
        mask_probs = mask_probs[0, 0]  # (128, 128)
        print(f"   After postprocess range: [{mask_probs.min():.4f}, {mask_probs.max():.4f}]")

        # Interpolate to original size
        pred_mask = F.interpolate(
            mask_probs.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        pred_mask_prob = pred_mask.cpu().numpy()[0, 0]

        print(f"   Final prob range: [{pred_mask_prob.min():.6f}, {pred_mask_prob.max():.6f}]")
        print(f"   Pixels > 0.5: {(pred_mask_prob > 0.5).sum()}")
        print(f"   Pixels > 0.1: {(pred_mask_prob > 0.1).sum()}")

        # Save probability mask
        prob_vis = (pred_mask_prob * 255).astype(np.uint8)
        Image.fromarray(prob_vis, mode="L").save(f"test_3d2d_prob_{prompt.replace(' ', '_')}.png")

        # Binary mask
        binary_mask = (pred_mask_prob > 0.5).astype(np.uint8)
        Image.fromarray(binary_mask * 255, mode="L").save(f"test_3d2d_binary_{prompt.replace(' ', '_')}.png")

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
