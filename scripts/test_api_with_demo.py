#!/usr/bin/env python3
"""
Test BiomedParse segmentation service using project demo data.

Tests:
1. Health check endpoint
2. Fundus optic disc segmentation
3. Fundus optic cup segmentation
4. Compare with ground truth masks (if available)
"""

import argparse
import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


DEMO_DIR = Path(__file__).parent / "biomedparse_datasets" / "BiomedParseData-Demo"
FUNDUS_IMAGE = DEMO_DIR / "demo" / "23_fundus_retinal.png"
GT_MASKS = {
    "optic disc": DEMO_DIR / "demo_mask" / "23_fundus_retinal_optic+disc.png",
    "optic cup": DEMO_DIR / "demo_mask" / "23_fundus_retinal_optic+cup.png",
}


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return intersection / union if union > 0 else 0.0


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    return 2 * intersection / total if total > 0 else 0.0


def test_health(base_url: str) -> bool:
    print("1. Health check...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        result = resp.json()
        print(f"   Status: {result['status']}")
        return result["status"] == "ok"
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def test_segmentation(
    base_url: str, image_path: Path, text_prompt: str, gt_mask_path: Path = None
) -> bool:
    print(f"2. Segmenting '{text_prompt}'...")
    print(f"   Image: {image_path.name}")

    image_b64 = encode_image_to_base64(image_path)
    print(f"   Image size: {len(image_b64) // 1024} KB (base64)")

    start = time.time()
    try:
        resp = requests.post(
            f"{base_url}/segment",
            json={"image": image_b64, "text_prompt": text_prompt},
            timeout=120,
        )
        elapsed = time.time() - start
        result = resp.json()
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    if not result["success"]:
        print(f"   FAILED: {result['error']}")
        return False

    # Decode predicted mask
    mask_bytes = base64.b64decode(result["mask"])
    pred_mask = np.array(Image.open(io.BytesIO(mask_bytes)))

    print(f"   Inference time: {elapsed:.2f}s")
    print(f"   Mask shape: {pred_mask.shape}")
    print(f"   Foreground pixels: {(pred_mask > 0).sum()}")

    # Save predicted mask
    output_path = f"test_api_mask_{text_prompt.replace(' ', '_')}.png"
    Image.fromarray(pred_mask, mode="L").save(output_path)
    print(f"   Saved to: {output_path}")

    # Compare with ground truth if available
    if gt_mask_path and gt_mask_path.exists():
        gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
        # Binarize gt (some gt masks may have values other than 0/255)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        pred_binary = (pred_mask > 127).astype(np.uint8)

        iou = compute_iou(pred_binary, gt_binary)
        dice = compute_dice(pred_binary, gt_binary)
        print(f"   IoU vs GT:  {iou:.4f}")
        print(f"   Dice vs GT: {dice:.4f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test BiomedParse API with demo data")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    parser.add_argument(
        "--prompt", default=None, help="Text prompt (default: test optic disc + optic cup)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BiomedParse Segmentation Service - Demo API Test")
    print("=" * 60)
    print()

    # Check demo files exist
    if not FUNDUS_IMAGE.exists():
        print(f"ERROR: Demo image not found: {FUNDUS_IMAGE}")
        print("Make sure git-lfs has pulled the demo data.")
        sys.exit(1)

    # Health check
    if not test_health(args.url):
        print("\nService is not healthy. Exiting.")
        sys.exit(1)
    print()

    # Run segmentation tests
    prompts = [args.prompt] if args.prompt else ["optic disc", "optic cup"]
    all_passed = True

    for i, prompt in enumerate(prompts, 2):
        gt_path = GT_MASKS.get(prompt)
        ok = test_segmentation(args.url, FUNDUS_IMAGE, prompt, gt_path)
        if not ok:
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("All API tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
