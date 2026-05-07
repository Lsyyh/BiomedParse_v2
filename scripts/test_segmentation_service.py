#!/usr/bin/env python3
"""
Test script for BiomedParse v2 Segmentation Service

Usage:
    python3 test_segmentation_service.py --image path/to/image.png --prompt "optic disc"
"""

import argparse
import base64
import io
import sys

import requests
from PIL import Image
import numpy as np


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_mask_from_base64(mask_base64: str) -> np.ndarray:
    """Decode base64 mask to numpy array."""
    mask_bytes = base64.b64decode(mask_base64)
    mask_image = Image.open(io.BytesIO(mask_bytes))
    return np.array(mask_image)


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        result = response.json()
        print(f"Health check: {result['status']}")
        return result['status'] == 'ok'
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_segmentation(base_url: str, image_path: str, text_prompt: str) -> bool:
    """Test segmentation endpoint."""
    try:
        # Encode image
        print(f"Encoding image: {image_path}")
        image_base64 = encode_image_to_base64(image_path)

        # Send request
        print(f"Sending segmentation request with prompt: '{text_prompt}'")
        response = requests.post(
            f"{base_url}/segment",
            json={
                "image": image_base64,
                "text_prompt": text_prompt
            },
            timeout=120  # Longer timeout for first request (model loading)
        )

        # Parse response
        result = response.json()

        if result['success']:
            print("Segmentation successful!")

            # Decode mask
            mask = decode_mask_from_base64(result['mask'])
            print(f"Mask shape: {mask.shape}")
            print(f"Mask unique values: {np.unique(mask)}")

            # Save mask
            output_path = f"test_mask_{text_prompt.replace(' ', '_')}.png"
            mask_image = Image.fromarray(mask, mode="L")
            mask_image.save(output_path)
            print(f"Mask saved to: {output_path}")

            return True
        else:
            print(f"Segmentation failed: {result['error']}")
            return False

    except Exception as e:
        print(f"Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test BiomedParse v2 Segmentation Service")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default="optic disc", help="Text prompt for segmentation")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    args = parser.parse_args()

    print("=" * 50)
    print("BiomedParse v2 Segmentation Service Test")
    print("=" * 50)
    print()

    # Test health
    print("1. Testing health endpoint...")
    if not test_health(args.url):
        print("Service is not healthy. Exiting.")
        sys.exit(1)
    print()

    # Test segmentation
    print("2. Testing segmentation endpoint...")
    if test_segmentation(args.url, args.image, args.prompt):
        print()
        print("All tests passed!")
    else:
        print()
        print("Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
