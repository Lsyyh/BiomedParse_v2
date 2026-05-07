<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

---

<a id="english"></a>

# BiomedParse API Service

> **Fork of [microsoft/BiomedParse v2](https://github.com/microsoft/BiomedParse)** — This project integrates both v1 (2D) and v2 (3D) model inference into a unified FastAPI REST API service for local deployment.

[[`Original Repo`](https://github.com/microsoft/BiomedParse)] [[`Paper`](https://aka.ms/biomedparse-paper)] [[`Model`](https://huggingface.co/microsoft/BiomedParse)]

## Overview

**BiomedParse** is a foundation model for biomedical image analysis supporting segmentation, detection, and recognition across nine imaging modalities (CT, MRI, Ultrasound, X-Ray, Pathology, Endoscopy, Dermoscopy, Fundus, OCT).

This fork discovers and fixes a critical compatibility issue when running v1 model inference within the v2 architecture, and provides a unified API service supporting both models.

## v1 vs v2: Key Differences

| Feature | v1 (2D) | v2 (3D) |
|---|---|---|
| Image type | 2D images | 3D volumes |
| Weights | `biomedparse_v1.pt` | `biomedparse_v2.ckpt` |
| Queries | 101 class-specific | 16 generic |
| Query matching | `vl_similarity` (vision-text) | `object_existence` gating |
| 2D Fundus segmentation | IoU > 0.95 | Poor (not target scenario) |
| 3D CT/MRI segmentation | Not supported | Native support |

### Critical Finding

The v1 model uses **101 class-specific queries**, each mapped to an anatomy. During inference, `vl_similarity` computes vision-text similarity per query and selects the best-matching query's mask. The v2 model uses **16 generic queries** with `object_existence` gating, optimized for 3D volumes.

When loading v1 weights into the v2 architecture, `vl_similarity` matching **must** be implemented. Without it, averaging all queries yields near-zero IoU (~0.0). With proper matching, IoU exceeds 0.95.

## Modifications

### 1. v1 Compatibility Layer

- **`src/model/transformer_decoder/boltzformer_cls_decoder.py`**: Modified `forward_prediction_heads` to return `class_embed` (visual embeddings) and added `pred_captions` to decoder output for `vl_similarity` matching.
- **`configs/model/biomedparse_v1_2d.yaml`**: New v1-specific 2D model config (`convolute_outputs: False`).
- **`configs/model/sem_seg_head/biomedparse_v1_maskformer_head.yaml`**: New v1 head config.
- **`configs/model/sem_seg_head/predictor/boltzformer_decoder_v1.yaml`**: New v1 decoder config (`num_queries: 101`, `pre_self_attention: true`).

### 2. API Service

- **`biomedparse_segmentation_service.py`**: FastAPI service loading both v1 and v2 models, providing three inference endpoints.

### 3. Weight Key Mapping

When loading v1 weights into v2 architecture:

```
lang_encoder.lang_encoder  →  language_encoder.encoder_transformer
lang_encoder               →  language_encoder
query_feat.weight          →  query_feat_.weight
query_embed.weight         →  query_embed_.weight
(skip biomed_encoder)
```

## Project Structure

```
BiomedParse-2/
├── biomedparse_segmentation_service.py   # FastAPI API service
├── inference.py                          # Original 3D inference script
├── process_2D.py                         # Original 2D processing
├── utils.py                              # Utilities
├── start_segmentation_service.sh         # Service startup script
├── check_deployment.sh                   # Deployment check script
├── configs/                              # Hydra configs
│   └── model/
│       ├── biomedparse.yaml              # v2 original 2D config
│       ├── biomedparse_3D.yaml           # v2 3D config
│       └── biomedparse_v1_2d.yaml        # v1 2D config (NEW)
├── src/                                  # Model source code
│   └── model/
│       ├── biomedparse.py                # 2D model
│       ├── biomedparse_3D.py             # 3D model
│       └── transformer_decoder/
│           └── boltzformer_cls_decoder.py # decoder (MODIFIED)
├── inference_utils/                      # Inference utilities
├── scripts/                              # Test & diagnostic scripts
│   ├── test_api_final.py                 # Comprehensive API test
│   ├── test_api_3d.py                    # 3D volume API test
│   └── ...                               # Other diagnostic scripts
├── examples/                             # Example data (.npz volumes + GT)
├── biomedparse_datasets/                 # Demo dataset
└── LICENSE                               # MIT License
```

## Deployment Guide

### Prerequisites

- CUDA GPU (16GB+ VRAM recommended)
- Conda
- HuggingFace account (must accept [model license](https://huggingface.co/microsoft/BiomedParse))

### 1. Environment Setup

```bash
conda create -n biomedparse_v2 python=3.10.14
conda activate biomedparse_v2

pip install -r assets/requirements/requirements.txt
pip install azureml-automl-core opencv-python
pip install git+https://github.com/facebookresearch/detectron2.git
pip install fastapi uvicorn requests

huggingface-cli login
```

### 2. Start Service

```bash
# Option 1: Use startup script
./start_segmentation_service.sh

# Option 2: Direct start
python biomedparse_segmentation_service.py
```

Service runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3. API Endpoints

#### `POST /segment` — 2D Image Segmentation

```json
{
    "image": "<base64 encoded image>",
    "text_prompt": "optic disc",
    "dimension": "2d"
}
```

- `dimension="2d"`: v1 model + `vl_similarity` matching (recommended for 2D)
- `dimension="3d"`: v2 model single-slice inference

#### `POST /segment_volume` — 3D Volume Segmentation

```json
{
    "volume": "<base64 encoded .npz file>"
}
```

The `.npz` file must contain `imgs` (volume data) and `text_prompts` (text prompt dict).

#### `GET /health` — Health Check

### 4. Test

```bash
python scripts/test_api_final.py    # 2D + 3D single-slice
python scripts/test_api_3d.py       # 3D volume inference
```

### 5. Python Example

```python
import requests, base64
from PIL import Image
import io

with open("fundus.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/segment", json={
    "image": image_b64, "text_prompt": "optic disc", "dimension": "2d"
})

mask = Image.open(io.BytesIO(base64.b64decode(resp.json()["mask"])))
mask.save("optic_disc_mask.png")
```

## Test Results

### 2D Segmentation (v1 Model)

| Test Case | IoU | Dice |
|---|---|---|
| optic disc | 0.9536 | 0.9763 |
| optic cup | 0.9545 | 0.9767 |
| left lung | 0.9743 | 0.9870 |

### 3D Volume Segmentation (v2 Model)

| Test Case | Status |
|---|---|
| CT_AMOS_amos_0018 | PASS |
| CT_AMOS_amos_0328 | PASS |
| CT_TotalSeg_bone_s0942 | PASS |
| MRI_Brain_tumor_001 | PASS |

## Citation

```bibtex
@article{zhao2025foundation,
  title={A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities},
  author={Zhao, Theodore and Gu, Yu and Yang, Jianwei and Usuyama, Naoto and Lee, Ho Hin and Kiblawi, Sid and Naumann, Tristan and Gao, Jianfeng and Crabtree, Angela and Abel, Jacob and others},
  journal={Nature methods},
  volume={22},
  number={1},
  pages={166--176},
  year={2025}
}

@inproceedings{zhao2025boltzmann,
  title={Boltzmann Attention Sampling for Image Analysis with Small Objects},
  author={Zhao, Theodore and Kiblawi, Sid and Usuyama, Naoto and Lee, Ho Hin and Preston, Sam and Poon, Hoifung and Wei, Mu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25950--25959},
  year={2025}
}
```

## License

MIT License. The model is for research and development use only, not for clinical decision-making.

---

<a id="中文"></a>

# BiomedParse API 分割服务

> **基于 [microsoft/BiomedParse v2](https://github.com/microsoft/BiomedParse) 的二次开发** — 集成 v1（2D）和 v2（3D）双模型推理能力，封装为 FastAPI REST API 服务，便于本地部署和外部系统集成。

[[`原项目`](https://github.com/microsoft/BiomedParse)] [[`论文`](https://aka.ms/biomedparse-paper)] [[`模型`](https://huggingface.co/microsoft/BiomedParse)]

## 项目简介

**BiomedParse** 是一个面向生物医学图像的基础模型，支持分割、检测和识别任务，覆盖 CT、MRI、超声、X 光、病理、内窥镜、皮肤镜、眼底、OCT 等九种成像模态。

本项目在原版 v2 的基础上，发现并修复了 v1 模型在 v2 架构下的推理兼容性问题，实现了同时支持 v1（2D）和 v2（3D）的统一 API 服务。

## v1 与 v2 版本差异

| 特性 | v1 (2D) | v2 (3D) |
|---|---|---|
| 适用图像 | 2D 图像 | 3D 体数据 |
| 模型权重 | `biomedparse_v1.pt` | `biomedparse_v2.ckpt` |
| Query 数量 | 101（类别特定） | 16（通用） |
| 查询匹配 | `vl_similarity` 视觉-文本相似度 | `object_existence` 置信度门控 |
| 2D 眼底分割 | IoU > 0.95 | 效果较差（非目标场景） |
| 3D CT/MRI 分割 | 不支持 | 原生支持 |

### 关键发现

v1 模型使用 **101 个类别特定的 query**，每个 query 对应一个解剖结构。推理时通过 `vl_similarity` 计算每个 query 的视觉嵌入与文本嵌入的相似度，选择最佳匹配 query 的 mask 作为输出。

v2 模型使用 **16 个通用 query**，配合 `object_existence` 置信度门控来过滤无效预测。这种设计针对 3D 体数据优化，在 2D 图像上的分割效果不如 v1。

将 v1 权重加载到 v2 架构时，**必须**实现 `vl_similarity` 匹配机制。否则对所有 query 取平均会导致分割效果极差（IoU ≈ 0），而正确实现后 IoU 可达 0.95 以上。

## 本项目修改

### 1. v1 模型兼容层

- **`src/model/transformer_decoder/boltzformer_cls_decoder.py`**：修改 `forward_prediction_heads` 返回 `class_embed`（视觉嵌入），并在 decoder 输出中增加 `pred_captions` 字段，用于 `vl_similarity` 匹配。
- **`configs/model/biomedparse_v1_2d.yaml`**：新增 v1 专用 2D 模型配置（`convolute_outputs: False`）。
- **`configs/model/sem_seg_head/biomedparse_v1_maskformer_head.yaml`**：新增 v1 专用 head 配置。
- **`configs/model/sem_seg_head/predictor/boltzformer_decoder_v1.yaml`**：新增 v1 decoder 配置（`num_queries: 101`, `pre_self_attention: true`）。

### 2. API 服务

- **`biomedparse_segmentation_service.py`**：FastAPI 服务，同时加载 v1 和 v2 模型，提供三个推理端点。

### 3. 权重名映射

v1 权重加载到 v2 架构时需要的 key 映射：

```
lang_encoder.lang_encoder  →  language_encoder.encoder_transformer
lang_encoder               →  language_encoder
query_feat.weight          →  query_feat_.weight
query_embed.weight         →  query_embed_.weight
（跳过 biomed_encoder）
```

## 项目结构

```
BiomedParse-2/
├── biomedparse_segmentation_service.py   # FastAPI API 服务
├── inference.py                          # 原始 3D 推理脚本
├── process_2D.py                         # 原始 2D 处理脚本
├── utils.py                              # 工具函数
├── start_segmentation_service.sh         # 服务启动脚本
├── check_deployment.sh                   # 部署检查脚本
├── configs/                              # Hydra 配置
│   └── model/
│       ├── biomedparse.yaml              # v2 原始 2D 配置
│       ├── biomedparse_3D.yaml           # v2 3D 配置
│       └── biomedparse_v1_2d.yaml        # v1 2D 配置（新增）
├── src/                                  # 模型源码
│   └── model/
│       ├── biomedparse.py                # 2D 模型
│       ├── biomedparse_3D.py             # 3D 模型
│       └── transformer_decoder/
│           └── boltzformer_cls_decoder.py # decoder（已修改）
├── inference_utils/                      # 推理工具
├── scripts/                              # 测试和诊断脚本
│   ├── test_api_final.py                 # API 综合测试
│   ├── test_api_3d.py                    # 3D 体数据 API 测试
│   └── ...                               # 其他诊断脚本
├── examples/                             # 示例数据（.npz 体数据 + GT）
├── biomedparse_datasets/                 # Demo 数据集
└── LICENSE                               # MIT License
```

## 部署指南

### 前置条件

- CUDA GPU（推荐 16GB+ 显存）
- Conda 环境
- HuggingFace 账号（需接受[模型协议](https://huggingface.co/microsoft/BiomedParse)）

### 1. 环境配置

```bash
conda create -n biomedparse_v2 python=3.10.14
conda activate biomedparse_v2

pip install -r assets/requirements/requirements.txt
pip install azureml-automl-core opencv-python
pip install git+https://github.com/facebookresearch/detectron2.git
pip install fastapi uvicorn requests

huggingface-cli login
```

### 2. 启动服务

```bash
# 方法 1: 使用启动脚本
./start_segmentation_service.sh

# 方法 2: 直接启动
python biomedparse_segmentation_service.py
```

服务默认运行在 `http://localhost:8000`，交互式文档访问 `http://localhost:8000/docs`。

### 3. API 端点

#### `POST /segment` — 2D 图像分割

```json
{
    "image": "<base64 编码的图像>",
    "text_prompt": "optic disc",
    "dimension": "2d"
}
```

- `dimension="2d"`：使用 v1 模型 + `vl_similarity` 匹配（推荐用于 2D 图像）
- `dimension="3d"`：使用 v2 模型单切片推理

#### `POST /segment_volume` — 3D 体数据分割

```json
{
    "volume": "<base64 编码的 .npz 文件>"
}
```

`.npz` 文件需包含 `imgs`（体数据）和 `text_prompts`（文本提示字典）。

#### `GET /health` — 健康检查

### 4. 测试服务

```bash
python scripts/test_api_final.py    # 2D + 3D 单切片测试
python scripts/test_api_3d.py       # 3D 体数据推理测试
```

### 5. Python 调用示例

```python
import requests, base64
from PIL import Image
import io

with open("fundus.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/segment", json={
    "image": image_b64, "text_prompt": "optic disc", "dimension": "2d"
})

mask = Image.open(io.BytesIO(base64.b64decode(resp.json()["mask"])))
mask.save("optic_disc_mask.png")
```

## 测试结果

### 2D 分割（v1 模型）

| 测试用例 | IoU | Dice |
|---|---|---|
| optic disc（视盘） | 0.9536 | 0.9763 |
| optic cup（视杯） | 0.9545 | 0.9767 |
| left lung（左肺） | 0.9743 | 0.9870 |

### 3D 体数据分割（v2 模型）

| 测试用例 | 状态 |
|---|---|
| CT_AMOS_amos_0018 | PASS |
| CT_AMOS_amos_0328 | PASS |
| CT_TotalSeg_bone_s0942 | PASS |
| MRI_Brain_tumor_001 | PASS |

## 引用

```bibtex
@article{zhao2025foundation,
  title={A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities},
  author={Zhao, Theodore and Gu, Yu and Yang, Jianwei and Usuyama, Naoto and Lee, Ho Hin and Kiblawi, Sid and Naumann, Tristan and Gao, Jianfeng and Crabtree, Angela and Abel, Jacob and others},
  journal={Nature methods},
  volume={22},
  number={1},
  pages={166--176},
  year={2025}
}

@inproceedings{zhao2025boltzmann,
  title={Boltzmann Attention Sampling for Image Analysis with Small Objects},
  author={Zhao, Theodore and Kiblawi, Sid and Usuyama, Naoto and Lee, Ho Hin and Preston, Sam and Poon, Hoifung and Wei, Mu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25950--25959},
  year={2025}
}
```

## 许可证

MIT License。模型仅限研究和开发用途，不适用于临床决策。
