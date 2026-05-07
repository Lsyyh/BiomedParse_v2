#!/usr/bin/env python3
"""
Test BiomedParse API 3D volume inference on all example volumes.

For each volume in examples/imgs/, sends full volume to API /segment_volume
endpoint and computes Dice/IoU against ground truth in examples/gts/.
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).parent
IMGS_DIR = PROJECT_ROOT / "examples" / "imgs"
GTS_DIR = PROJECT_ROOT / "examples" / "gts"


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    return 2 * intersection / total if total > 0 else float("nan")


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / union if union > 0 else float("nan")


def main():
    print("=" * 70)
    print("BiomedParse API - 3D Volume Inference Test")
    print("=" * 70)
    print()

    base_url = "http://localhost:8000"

    # Health check
    print("[1] Health check...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        print(f"  {resp.json()['status']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
    print()

    # Collect volumes
    npz_files = sorted(IMGS_DIR.glob("*.npz"))
    print(f"[2] Testing {len(npz_files)} volumes via API /segment_volume")
    print("-" * 70)

    results = []

    for npz_path in npz_files:
        case_name = npz_path.stem
        gt_path = GTS_DIR / npz_path.name

        print(f"\n  Case: {case_name}")

        # Load volume as base64
        with open(npz_path, "rb") as f:
            volume_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Show volume info
        data = np.load(str(npz_path), allow_pickle=True)
        imgs = data["imgs"]
        text_prompts = data["text_prompts"].item()
        ids = sorted([int(k) for k in text_prompts.keys() if k != "instance_label"])
        class_names = [text_prompts[str(i)] for i in ids]
        print(f"  Volume: {imgs.shape} | Classes: {class_names}")

        # Call API
        start = time.time()
        try:
            resp = requests.post(
                f"{base_url}/segment_volume",
                json={"volume": volume_b64},
                timeout=300,
            )
            elapsed = time.time() - start
            result = resp.json()
        except Exception as e:
            print(f"  API error: {e}")
            results.append({"case": case_name, "pass": False, "error": str(e)})
            continue

        if not result["success"]:
            print(f"  Inference failed: {result['error']}")
            results.append({"case": case_name, "pass": False, "error": result["error"]})
            continue

        # Decode prediction
        seg_bytes = base64.b64decode(result["segmentation"])
        seg_data = np.load(io.BytesIO(seg_bytes))
        pred_segs = seg_data["segs"]
        print(f"  Inference: {elapsed:.2f}s | Output shape: {pred_segs.shape}")

        # Compare with GT
        case_result = {"case": case_name, "time": elapsed, "pass": True, "classes": []}

        if gt_path.exists():
            gt_data = np.load(str(gt_path), allow_pickle=True)
            gt_segs = gt_data["gts"]

            for class_id in ids:
                gt_class = (gt_segs == class_id)
                pred_class = (pred_segs == class_id)

                dice = compute_dice(pred_class, gt_class)
                iou = compute_iou(pred_class, gt_class)
                class_name = text_prompts[str(class_id)]

                dice_val = float(dice) if not np.isnan(dice) else 0.0
                iou_val = float(iou) if not np.isnan(iou) else 0.0

                print(f"  [{class_name}] Dice={dice_val:.4f} IoU={iou_val:.4f}")
                case_result["classes"].append({
                    "name": class_name,
                    "dice": dice_val,
                    "iou": iou_val,
                })
        else:
            print(f"  No GT found")

        results.append(case_result)

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"{'Case':<45} {'Class':<20} {'Dice':>8} {'IoU':>8} {'Time':>8}")
    print("-" * 90)

    for r in results:
        if "error" in r:
            print(f"{r['case']:<45} {'ERROR':<20} {'N/A':>8} {'N/A':>8}")
            continue
        if r.get("classes"):
            for i, c in enumerate(r["classes"]):
                case_label = r["case"] if i == 0 else ""
                time_label = f"{r['time']:.1f}s" if i == 0 else ""
                print(f"{case_label:<45} {c['name']:<20} {c['dice']:>8.4f} {c['iou']:>8.4f} {time_label:>8}")
        else:
            print(f"{r['case']:<45} {'(no GT)':<20} {'N/A':>8} {'N/A':>8} {r['time']:>6.1f}s")

    # Average
    all_dices = [c["dice"] for r in results for c in r.get("classes", [])]
    all_ious = [c["iou"] for r in results for c in r.get("classes", [])]
    if all_dices:
        print("-" * 90)
        print(f"{'Average':<45} {'':<20} {np.mean(all_dices):>8.4f} {np.mean(all_ious):>8.4f}")

    print()
    all_pass = all(r.get("pass", False) for r in results)
    print("All tests PASSED!" if all_pass else "Some tests FAILED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
