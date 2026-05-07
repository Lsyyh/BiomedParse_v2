#!/usr/bin/env python3
"""
Direct 3D model diagnostic test.

Tests BiomedParse v2 3D model on example CT data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python direct_model_test_3D.py
"""

import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download
from utils import process_input, process_output, slice_nms


def postprocess(model_outputs, object_existence, threshold=0.5, do_nms=True):
    if do_nms and model_outputs.shape[0] > 1:
        return slice_nms(model_outputs.sigmoid(), object_existence.sigmoid(),
                         iou_threshold=0.5, score_threshold=threshold)
    mask = (model_outputs.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)
    return mask


def merge_multiclass_masks(masks, ids):
    bg_mask = 0.5 * torch.ones_like(masks[0:1])
    keep_masks = torch.cat([bg_mask, masks], dim=0)
    class_mask = keep_masks.argmax(dim=0)
    id_map = {j + 1: int(ids[j]) for j in range(len(ids)) if j + 1 != int(ids[j])}
    if len(id_map) > 0:
        orig_mask = class_mask.clone()
        for j in id_map:
            class_mask[orig_mask == j] = id_map[j]
    return class_mask


def compute_dice(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return float('nan')
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def main():
    print("=" * 60)
    print("BiomedParse v2 3D Model Diagnostic Test")
    print("=" * 60)
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print()

    # Load 3D model
    print("1. Loading 3D model...")
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="diagnostic_3d")
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

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
    if missing:
        print(f"   First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"   First 5 unexpected: {unexpected[:5]}")
    model = model.to(device).eval()
    print("   3D Model loaded!")
    print()

    # Test cases
    test_cases = [
        ("CT_AMOS_amos_0018", "examples/imgs/CT_AMOS_amos_0018.npz", "examples/gts/CT_AMOS_amos_0018.npz"),
    ]

    for name, img_path, gt_path in test_cases:
        print(f"2. Testing: {name}")
        npz_data = np.load(img_path, allow_pickle=True)
        imgs = npz_data["imgs"]
        text_prompts = npz_data["text_prompts"].item()

        ids = [int(_) for _ in text_prompts.keys() if _ != "instance_label"]
        ids.sort()
        text = "[SEP]".join([text_prompts[str(i)] for i in ids])

        print(f"   Image shape: {imgs.shape}")
        print(f"   Number of classes: {len(ids)}")
        print(f"   Text prompt (first 100 chars): {text[:100]}...")

        # Process input
        imgs_processed, pad_width, padded_size, valid_axis = process_input(imgs, 512)
        imgs_processed = imgs_processed.to(device).int()

        input_tensor = {
            "image": imgs_processed.unsqueeze(0),
            "text": [text],
        }

        print(f"   Processed image shape: {imgs_processed.shape}")

        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=4)

        mask_preds = output["predictions"]["pred_gmasks"]
        print(f"   Raw mask_preds shape: {mask_preds.shape}")
        print(f"   Raw mask_preds range: [{mask_preds.min():.4f}, {mask_preds.max():.4f}]")

        object_existence = output["predictions"]["object_existence"]
        print(f"   object_existence shape: {object_existence.shape}")
        print(f"   object_existence range: [{object_existence.min():.4f}, {object_existence.max():.4f}]")

        # Interpolate
        mask_preds = F.interpolate(
            mask_preds, size=(512, 512), mode="bicubic", align_corners=False, antialias=True
        )

        # Postprocess
        mask_preds = postprocess(mask_preds, object_existence)
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        print(f"   Final mask shape: {mask_preds.shape}")
        print(f"   Final mask unique values: {np.unique(mask_preds)}")

        # Compare with ground truth
        if os.path.exists(gt_path):
            gt_data = np.load(gt_path, allow_pickle=True)
            gt = gt_data["gts"]
            print(f"   GT shape: {gt.shape}")

            # Compute Dice for each class
            for cls_id in ids:
                pred_cls = (mask_preds == cls_id).astype(np.uint8)
                gt_cls = (gt == cls_id).astype(np.uint8)
                dice = compute_dice(gt_cls, pred_cls)
                gt_pixels = gt_cls.sum()
                pred_pixels = pred_cls.sum()
                print(f"   Class {cls_id} ({text_prompts[str(cls_id)][:30]}): Dice={dice:.4f}, GT_pix={gt_pixels}, Pred_pix={pred_pixels}")

        # Cleanup
        del imgs_processed, input_tensor, output, mask_preds
        gc.collect()
        torch.cuda.empty_cache()
        print()

    print("=" * 60)
    print("3D Diagnostic test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
