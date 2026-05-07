#!/usr/bin/env python3
"""
Final API test for BiomedParse segmentation service (v1 2D + v2 3D).

Tests:
1. Health check
2. 2D fundus: optic disc segmentation (v1 + vl_similarity)
3. 2D fundus: optic cup segmentation (v1 + vl_similarity)
4. 2D chest X-ray: lung segmentation
5. 3D CT: organ segmentation (v2)
6. Compare with ground truth masks
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


PROJECT_ROOT = Path(__file__).parent
DEMO_DIR = PROJECT_ROOT / "biomedparse_datasets" / "BiomedParseData-Demo"
DEMO_IMG_DIR = DEMO_DIR / "demo"
DEMO_MASK_DIR = DEMO_DIR / "demo_mask"
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "imgs"

# Test cases
TESTS_2D = [
    {
        "image": "23_fundus_retinal.png",
        "prompt": "optic disc",
        "gt_mask": "23_fundus_retinal_optic+disc.png",
        "min_iou": 0.80,
    },
    {
        "image": "23_fundus_retinal.png",
        "prompt": "optic cup",
        "gt_mask": "23_fundus_retinal_optic+cup.png",
        "min_iou": 0.80,
    },
    {
        "image": "15_X-Ray_chest.png",
        "prompt": "left lung",
        "gt_mask": "15_X-Ray_chest_left+lung.png",
        "min_iou": 0.50,
    },
]

TESTS_3D = [
    {
        "image": "CT_AMOS_amos_0018.npz",
        "prompt": "liver",
    },
]


def encode_image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_npz_as_base64(path: Path) -> str:
    """Load .npz volume, take middle slice, encode as PNG base64."""
    data = np.load(path)
    key = list(data.keys())[0]
    volume = data[key]
    # Take middle slice
    mid = volume.shape[0] // 2
    slice_2d = volume[mid]
    # Normalize to 0-255 uint8
    slice_norm = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(slice_norm, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return intersection / union if union > 0 else 0.0


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    return 2 * intersection / total if total > 0 else 0.0


def test_health(base_url: str) -> bool:
    print("[1/5] Health check...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        result = resp.json()
        print(f"  Status: {result['status']}")
        return "2D(v1): loaded" in result["status"] and "3D(v2): loaded" in result["status"]
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_2d_segmentation(base_url: str, tc: dict) -> dict:
    """Test 2D segmentation via API. Returns result dict."""
    image_path = DEMO_IMG_DIR / tc["image"]
    gt_path = DEMO_MASK_DIR / tc["gt_mask"]

    print(f"  Image: {tc['image']}")
    print(f"  Prompt: '{tc['prompt']}'")

    image_b64 = encode_image_to_base64(image_path)

    start = time.time()
    resp = requests.post(
        f"{base_url}/segment",
        json={
            "image": image_b64,
            "text_prompt": tc["prompt"],
            "dimension": "2d",
        },
        timeout=120,
    )
    elapsed = time.time() - start
    result = resp.json()

    if not result["success"]:
        print(f"  FAILED: {result['error']}")
        return {"pass": False, "error": result["error"]}

    # Decode predicted mask
    mask_bytes = base64.b64decode(result["mask"])
    pred_mask = np.array(Image.open(io.BytesIO(mask_bytes)))

    fg_pixels = (pred_mask > 0).sum()
    print(f"  Inference: {elapsed:.2f}s | Foreground pixels: {fg_pixels}")

    # Save
    out_name = f"test_api_2d_{tc['prompt'].replace(' ', '_')}.png"
    Image.fromarray(pred_mask, mode="L").save(out_name)

    # Compare with GT
    iou = dice = 0.0
    if gt_path.exists():
        gt_mask = np.array(Image.open(gt_path).convert("L"))
        gt_binary = (gt_mask > 127).astype(np.uint8)
        pred_binary = (pred_mask > 127).astype(np.uint8)
        iou = compute_iou(pred_binary, gt_binary)
        dice = compute_dice(pred_binary, gt_binary)
        print(f"  IoU={iou:.4f} Dice={dice:.4f} (threshold: min_iou={tc['min_iou']})")
    else:
        print(f"  GT not found: {gt_path}")

    passed = iou >= tc["min_iou"]
    return {"pass": passed, "iou": iou, "dice": dice, "time": elapsed}


def test_3d_segmentation(base_url: str, tc: dict) -> dict:
    """Test 3D segmentation via API (single slice)."""
    image_path = EXAMPLES_DIR / tc["image"]

    print(f"  Volume: {tc['image']} (middle slice)")
    print(f"  Prompt: '{tc['prompt']}'")

    image_b64 = load_npz_as_base64(image_path)

    start = time.time()
    resp = requests.post(
        f"{base_url}/segment",
        json={
            "image": image_b64,
            "text_prompt": tc["prompt"],
            "dimension": "3d",
        },
        timeout=120,
    )
    elapsed = time.time() - start
    result = resp.json()

    if not result["success"]:
        print(f"  FAILED: {result['error']}")
        return {"pass": False, "error": result["error"]}

    mask_bytes = base64.b64decode(result["mask"])
    pred_mask = np.array(Image.open(io.BytesIO(mask_bytes)))

    fg_pixels = (pred_mask > 0).sum()
    print(f"  Inference: {elapsed:.2f}s | Foreground pixels: {fg_pixels}")

    out_name = f"test_api_3d_{tc['prompt'].replace(' ', '_')}.png"
    Image.fromarray(pred_mask, mode="L").save(out_name)

    # 3D has no simple GT for single-slice, just check non-empty
    passed = fg_pixels > 0
    return {"pass": passed, "fg_pixels": fg_pixels, "time": elapsed}


def main():
    parser = argparse.ArgumentParser(description="BiomedParse API final test")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    args = parser.parse_args()

    print("=" * 60)
    print("BiomedParse Segmentation Service - Final API Test")
    print("=" * 60)
    print()

    # Health
    if not test_health(args.url):
        print("\nService not healthy. Exiting.")
        sys.exit(1)
    print()

    all_passed = True

    # 2D tests
    print("[2/5] 2D Segmentation Tests (v1 model)")
    print("-" * 40)
    for tc in TESTS_2D:
        r = test_2d_segmentation(args.url, tc)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  Result: {status}")
        if not r["pass"]:
            all_passed = False
        print()

    # 3D tests
    print("[3/5] 3D Segmentation Tests (v2 model)")
    print("-" * 40)
    for tc in TESTS_3D:
        r = test_3d_segmentation(args.url, tc)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  Result: {status}")
        if not r["pass"]:
            all_passed = False
        print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("All API tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
