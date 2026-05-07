# BiomedParse API - MedAgent-Ultra Integration Guide

> This document describes how to deploy and use the BiomedParse segmentation API service for integration with MedAgent-Ultra.

## Deployment

### GPU Selection

The BiomedParse service loads two models simultaneously (v1 for 2D, v2 for 3D), requiring **at least 16GB VRAM**. When deploying:

- **Single GPU**: The service automatically uses `cuda:0`. Ensure no other heavy processes are using the GPU.
- **Multi-GPU**: Set `CUDA_VISIBLE_DEVICES` to select a specific GPU before starting the service:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python biomedparse_segmentation_service.py
  ```
- **No GPU**: The service falls back to CPU, but inference will be very slow (not recommended).

### Starting the Service

```bash
cd /path/to/BiomedParse-2
conda activate biomedparse_v2

# Optional: select GPU
export CUDA_VISIBLE_DEVICES=0

# Start service (default port 8000)
python biomedparse_segmentation_service.py
```

The service will:
1. Download model weights from HuggingFace on first run (requires `huggingface-cli login`)
2. Load both v1 (2D) and v2 (3D) models into GPU memory
3. Listen on `http://0.0.0.0:8000`

Wait for the log message `Application startup complete` before sending requests.

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "2D(v1): loaded, 3D(v2): loaded"}
```

## API Usage

### 2D Image Segmentation

For 2D images (fundus, pathology, X-ray, etc.), use the **v1 model** with `dimension="2d"`:

```python
import requests
import base64

def segment_2d(image_path: str, text_prompt: str) -> bytes:
    """Segment a 2D image using BiomedParse v1 model.

    Args:
        image_path: Path to the image file (PNG, JPG, etc.)
        text_prompt: Text description of the target structure
            Examples: "optic disc", "optic cup", "left lung", "neoplastic cells"

    Returns:
        Binary mask as bytes (PNG format, 0=background, 255=foreground)
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        "http://localhost:8000/segment",
        json={
            "image": image_b64,
            "text_prompt": text_prompt,
            "dimension": "2d",
        },
        timeout=120,
    )

    result = resp.json()
    if not result["success"]:
        raise RuntimeError(f"Segmentation failed: {result['error']}")

    return base64.b64decode(result["mask"])
```

### 3D Volume Segmentation

For 3D volumes (CT, MRI), use the `/segment_volume` endpoint with the **v2 model**:

```python
import requests
import base64
import numpy as np

def segment_3d_volume(npz_path: str) -> np.ndarray:
    """Segment a 3D volume using BiomedParse v2 model.

    Args:
        npz_path: Path to .npz file containing:
            - 'imgs': 3D numpy array (D, H, W) with intensity values in [0, 255]
            - 'text_prompts': dict mapping class IDs to text prompts
                Example: {"1": "liver", "2": "spleen"}

    Returns:
        3D numpy array of predicted segmentation labels (same shape as input)
    """
    with open(npz_path, "rb") as f:
        volume_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        "http://localhost:8000/segment_volume",
        json={"volume": volume_b64},
        timeout=600,  # 3D inference can take several minutes
    )

    result = resp.json()
    if not result["success"]:
        raise RuntimeError(f"3D segmentation failed: {result['error']}")

    import io
    seg_bytes = base64.b64decode(result["segmentation"])
    seg_data = np.load(io.BytesIO(seg_bytes))
    return seg_data["segs"]
```

### 3D Volume Input Format

The `.npz` file must contain:

| Key | Type | Description |
|-----|------|-------------|
| `imgs` | `np.ndarray` | 3D volume, shape `(D, H, W)`, intensity range `[0, 255]` |
| `text_prompts` | `dict` | Class ID to text prompt mapping, e.g. `{"1": "liver", "2": "spleen"}` |

**Preprocessing requirements for CT images:**
- Soft tissues: window width=400, level=40
- Lung: W=1500, L=-160
- Brain: W=80, L=40
- Bone: W=1800, L=400

For other modalities, clip intensity to 0.5th-99.5th percentile, then rescale to [0, 255].

## Model Selection Guide

| Input Type | Endpoint | Model | When to Use |
|---|---|---|---|
| 2D image (fundus, pathology, X-ray, etc.) | `POST /segment` with `dimension="2d"` | v1 | Default for all 2D images |
| 2D image (quick test) | `POST /segment` with `dimension="3d"` | v2 | Single-slice 3D model test |
| 3D volume (CT, MRI) | `POST /segment_volume` | v2 | Full volumetric segmentation |

## Supported Text Prompts

The model supports free-text prompts. Common examples by modality:

- **Fundus**: `optic disc`, `optic cup`, `retinal vessel`
- **Chest X-ray**: `left lung`, `right lung`, `lung`
- **CT Abdomen**: `liver`, `spleen`, `kidney`, `pancreas`, `tumor`
- **CT Chest**: `nodule`, `COVID-19 infection`, `tumor`
- **MRI Brain**: `edema`, `tumor core`, `whole tumor`, `enhancing tumor`
- **Pathology**: `neoplastic cells`, `inflammatory cells`, `epithelial cells`
- **Endoscopy**: `polyp`, `neoplastic polyp`
- **Dermoscopy**: `lesion`, `melanoma`
- **OCT**: `edema`

## Error Handling

```python
result = resp.json()
if not result["success"]:
    print(f"Error: {result['error']}")
    # Common errors:
    # - "v1 2D model not loaded" → model weights not downloaded
    # - "CUDA out of memory" → GPU VRAM insufficient
    # - "Failed to decode image" → invalid base64 encoding
```

## Performance Notes

- First request is slow (~30s) due to model warm-up
- Subsequent 2D requests: ~1-3 seconds per image
- 3D volume inference: ~30-120 seconds depending on volume size
- The service handles one request at a time (single worker)
- GPU memory: ~8-12GB for both models loaded simultaneously

## Troubleshooting

| Issue | Solution |
|---|---|
| `CUDA out of memory` | Close other GPU processes, or use a GPU with 16GB+ VRAM |
| `model not loaded` | Check HuggingFace login: `huggingface-cli whoami` |
| `Connection refused` | Ensure service is running: `curl http://localhost:8000/health` |
| Slow inference | First request is slow; subsequent requests should be faster |
| Poor 2D segmentation | Ensure `dimension="2d"` is set for 2D images |
