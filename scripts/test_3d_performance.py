#!/usr/bin/env python3
"""
Test BiomedParse v2 3D inference performance on all example volumes.

For each volume in examples/imgs/, runs full 3D inference and computes
Dice coefficient against ground truth in examples/gts/.
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


def load_case(file_path):
    """Load 3D volume, text prompts, and GT from npz file."""
    data = np.load(file_path, allow_pickle=True)
    image = data["imgs"]
    text_prompts = data["text_prompts"].item()
    gt = data["gts"] if "gts" in data else None
    return image, text_prompts, gt


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


def postprocess(model_outputs, object_existence, threshold=0.5, do_nms=True):
    """Postprocess model outputs with sigmoid, gating, and NMS."""
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


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute Dice coefficient between two binary masks."""
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def compute_iou(mask_gt, mask_pred):
    """Compute IoU between two binary masks."""
    intersection = (mask_gt & mask_pred).sum()
    union = (mask_gt | mask_pred).sum()
    if union == 0:
        return np.NaN
    return intersection / union


@torch.no_grad()
def run_3d_inference(model, npz_path, device):
    """
    Run full 3D inference on a volume.

    Returns:
        pred_segs: numpy array of predicted segmentation (same shape as input volume)
        text_prompts: dict of text prompts
        gt: ground truth volume (or None)
    """
    imgs, text_prompts, gt = load_case(str(npz_path))

    ids = [int(_) for _ in text_prompts.keys() if _ != "instance_label"]
    ids.sort()
    text = "[SEP]".join([text_prompts[str(i)] for i in ids])

    imgs_padded, pad_width, padded_size, valid_axis = process_input(imgs, 512)
    imgs_tensor = imgs_padded.to(device).int()

    input_tensor = {
        "image": imgs_tensor.unsqueeze(0),
        "text": [text],
    }

    output = model(input_tensor, mode="eval", slice_batch_size=4)

    mask_preds = output["predictions"]["pred_gmasks"]
    mask_preds = F.interpolate(
        mask_preds,
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
    mask_preds = merge_multiclass_masks(mask_preds, ids)
    pred_segs = process_output(mask_preds, pad_width, padded_size, valid_axis)

    # Cleanup
    del imgs_tensor, input_tensor, output, mask_preds
    gc.collect()
    torch.cuda.empty_cache()

    return pred_segs, text_prompts, gt


def main():
    print("=" * 70)
    print("BiomedParse v2 - 3D Inference Performance Test")
    print("=" * 70)
    print()

    imgs_dir = Path(PROJECT_ROOT) / "examples" / "imgs"
    gts_dir = Path(PROJECT_ROOT) / "examples" / "gts"

    # Collect all .npz files
    npz_files = sorted(imgs_dir.glob("*.npz"))
    if not npz_files:
        print("ERROR: No .npz files found in examples/imgs/")
        sys.exit(1)

    print(f"Found {len(npz_files)} volumes to test:")
    for f in npz_files:
        print(f"  - {f.name}")
    print()

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print()

    print("Loading v2 3D model...")
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs/model", job_name="test_3d_perf")
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

    v2_path = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v2.ckpt")
    v2_ckpt = torch.load(v2_path, map_location="cpu")

    if isinstance(v2_ckpt, dict):
        state_dict = v2_ckpt.get("state_dict", v2_ckpt.get("model", v2_ckpt))
    else:
        state_dict = v2_ckpt

    state_dict = {
        k[6:] if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"Model loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
    print()

    # Run inference on each volume
    all_results = []

    for npz_path in npz_files:
        case_name = npz_path.stem
        gt_path = gts_dir / npz_path.name

        print(f"{'=' * 70}")
        print(f"Case: {case_name}")
        print(f"{'=' * 70}")

        # Load to show info
        data = np.load(str(npz_path), allow_pickle=True)
        imgs = data["imgs"]
        text_prompts = data["text_prompts"].item()
        ids = sorted([int(k) for k in text_prompts.keys() if k != "instance_label"])
        class_names = [text_prompts[str(i)] for i in ids]
        print(f"  Volume shape: {imgs.shape}")
        print(f"  Classes ({len(ids)}): {class_names}")

        # Run inference
        import time
        start = time.time()
        pred_segs, text_prompts_out, _ = run_3d_inference(model, npz_path, device)
        elapsed = time.time() - start
        print(f"  Inference time: {elapsed:.2f}s")
        print(f"  Prediction shape: {pred_segs.shape}")

        # Compute metrics against GT
        if gt_path.exists():
            gt_data = np.load(str(gt_path), allow_pickle=True)
            gt_segs = gt_data["gts"]
            print(f"  GT shape: {gt_segs.shape}")

            # Per-class Dice
            case_results = {"case": case_name, "time": elapsed, "classes": []}
            for idx, class_id in enumerate(ids):
                gt_class = (gt_segs == class_id)
                pred_class = (pred_segs == class_id)

                dice = compute_dice_coefficient(gt_class, pred_class)
                iou = compute_iou(gt_class, pred_class)
                gt_voxels = gt_class.sum()
                pred_voxels = pred_class.sum()

                class_name = text_prompts[str(class_id)]
                print(f"  [{class_name}]")
                print(f"    GT voxels: {gt_voxels} | Pred voxels: {pred_voxels}")
                print(f"    Dice: {dice:.4f} | IoU: {iou:.4f}")

                case_results["classes"].append({
                    "name": class_name,
                    "dice": float(dice) if not np.isnan(dice) else 0.0,
                    "iou": float(iou) if not np.isnan(iou) else 0.0,
                })

            all_results.append(case_results)
        else:
            print(f"  GT file not found: {gt_path}")
            print("  Skipping Dice/IoU computation")
            all_results.append({"case": case_name, "time": elapsed, "classes": []})

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"{'Case':<45} {'Class':<25} {'Dice':>8} {'IoU':>8} {'Time':>8}")
    print("-" * 95)

    for r in all_results:
        if r["classes"]:
            for i, c in enumerate(r["classes"]):
                case_label = r["case"] if i == 0 else ""
                time_label = f"{r['time']:.1f}s" if i == 0 else ""
                print(f"{case_label:<45} {c['name']:<25} {c['dice']:>8.4f} {c['iou']:>8.4f} {time_label:>8}")
        else:
            print(f"{r['case']:<45} {'(no GT)':<25} {'N/A':>8} {'N/A':>8} {r['time']:>6.1f}s")

    # Overall average
    all_dices = [c["dice"] for r in all_results for c in r["classes"]]
    all_ious = [c["iou"] for r in all_results for c in r["classes"]]
    if all_dices:
        print("-" * 95)
        print(f"{'Average':<45} {'':<25} {np.mean(all_dices):>8.4f} {np.mean(all_ious):>8.4f}")

    print()
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
