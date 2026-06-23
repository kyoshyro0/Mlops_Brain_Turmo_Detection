# 🧠 Brain Tumor Detection — MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv11s-Ultralytics-00FFFF?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.6-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hệ thống **MLOps end-to-end** phát hiện khối u não (glioma, meningioma, pituitary) trên ảnh MRI sử dụng **YOLOv11s**, tích hợp đầy đủ quy trình từ **huấn luyện → đánh giá → triển khai → giám sát** với khả năng chuyển đổi runtime **PyTorch/ONNX** trực tiếp từ giao diện người dùng.

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Tính Năng Chính](#-tính-năng-chính)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt & Khởi Chạy](#-cài-đặt--khởi-chạy)
- [Pipeline MLOps](#-pipeline-mlops)
- [Model Serving](#-model-serving)
- [API Reference](#-api-reference)
- [Cấu Hình](#-cấu-hình)
- [Hiệu Năng Mô Hình](#-hiệu-năng-mô-hình)
- [Docker Deployment](#-docker-deployment)
- [Phát Triển & Kiểm Thử](#-phát-triển--kiểm-thử)
- [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)
- [Giấy Phép](#-giấy-phép)

---

## 🎯 Tổng Quan

### Bài toán

Phát hiện tự động khối u não trên ảnh chụp MRI, phân loại thành **4 lớp**:

| Lớp | Mô tả |
|-----|--------|
| `glioma` | U thần kinh đệm |
| `meningioma` | U màng não |
| `pituitary` | U tuyến yên |
| `notumor` | Không có khối u |

### Giải pháp

Một hệ thống MLOps production-grade xây dựng trên **YOLOv11s** (Ultralytics), tích hợp:

- **Registry-first model loading** — mô hình được phục vụ trực tiếp từ MLflow Model Registry qua custom PyFunc wrapper, không cần tải artifact thủ công
- **Zero-overhead backend switching** — chuyển đổi giữa PyTorch (GPU) và ONNX (CPU) ngay từ giao diện Streamlit mà không cần restart server
- **4-tier model fallback** — đảm bảo API luôn khởi động được ngay cả khi Registry không khả dụng
- **Async inference logging** — ghi log metric inference vào MLflow qua background thread, giảm thiểu latency response
- **LRU prediction cache** — cache kết quả dự đoán dựa trên hash nội dung ảnh + cấu hình, tránh tính toán lại

---

## 🏗 Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                          │
│   ┌──────────────────┐  ┌────────────────────────────────────┐  │
│   │  PyTorch / ONNX  │  │     Model Status Display           │  │
│   │     Toggle       │  │  (source, backend, availability)   │  │
│   └────────┬─────────┘  └────────────────────────────────────┘  │
├────────────┼────────────────────────────────────────────────────┤
│            ▼              FastAPI Backend                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PredictionService                                       │   │
│  │   1. Decode image (JPEG/PNG → NumPy RGB)                 │   │
│  │   2. LRU cache lookup (MD5 hash + config key)            │   │
│  │   3. Inference via YOLOWrapper                           │   │
│  │   4. Format results → JSON                               │   │
│  │   5. Async MLflow log (background thread)                │   │
│  │   6. Cache result + respond                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│            │                                                     │
│  ┌─────────▼────────────────────────────────────────────────┐   │
│  │  YOLOWrapper (mlflow.pyfunc.PythonModel)                 │   │
│  │   ┌──────────────┐   ┌────────────────────────────────┐  │   │
│  │   │  PyTorch      │   │  ONNX (lazy — loads on first  │  │   │
│  │   │  (always)     │   │  ONNX request only)           │  │   │
│  │   └──────────────┘   └────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ModelService — 4-tier fallback:                                 │
│   0. Registry @production  →  1. Registry latest version         │
│   2. deployed_models/vN/   →  3. Default yolo11s.pt              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Tính Năng Chính

### 🔄 Pipeline tự động hóa

- **5-step pipeline**: Train → Validate → Export ONNX → Save locally (versioned) → Register to MLflow
- Mỗi lần chạy pipeline tạo ra thư mục phiên bản mới `deployed_models/vN/` chứa cả `best.pt` và `best.onnx`
- Tự động gán alias `staging` cho model mới, hỗ trợ promote lên `production` qua CLI

### 📦 Model Registry

- Custom **YOLOWrapper** kế thừa `mlflow.pyfunc.PythonModel` — cho phép load trực tiếp qua `mlflow.pyfunc.load_model()`
- Bundle cả PyTorch và ONNX weights trong cùng một model version
- Lazy-load ONNX — zero extra RAM cho đến khi người dùng thực sự bật ONNX toggle

### 🚀 Serving Layer

- **FastAPI** backend với Swagger UI tự động (`/docs`)
- **Streamlit** frontend cho phép upload ảnh MRI, xem kết quả phát hiện, chuyển đổi backend
- CORS middleware cho phép tích hợp với bất kỳ frontend nào
- Prediction cache với LRU eviction (mặc định 50 entries)

### 📊 Experiment Tracking

- MLflow tracking cho training, evaluation, inference, deployment, monitoring
- Async inference logging — mọi request prediction đều được log mà không ảnh hưởng latency
- Tự động log metrics: processing time, số detections, confidence trung bình/cao nhất

### 🐳 Docker Support

- Multi-stage Dockerfile với 3 stages: `backend-builder`, `backend`, `frontend`
- Backend sử dụng NVIDIA CUDA 11.8 runtime image với GPU passthrough
- Frontend chạy trên `python:3.12-slim`, chỉ cài Streamlit + requests + Pillow (~300MB)
- Docker Compose orchestration cho cả 2 services

### 🧪 Testing & CI/CD

- Test suite phân tầng: unit, integration, end-to-end
- GitHub Actions workflow: lint → test → Docker build → container health check
- DVC tích hợp cho data versioning

---

## 📁 Cấu Trúc Dự Án

```
mlops_brain_turmo/
├── configs/                        # Cấu hình dataset
│   ├── data.yaml                   #   Training dataset config (4 classes)
│   └── data_test.yaml              #   Test dataset config
├── datasets/                       # Image datasets (không track trong git)
├── deployed_models/                # Model cache phiên bản hóa
│   └── v1/
│       ├── best.pt                 #   PyTorch weights
│       └── best.onnx               #   ONNX weights
├── scripts/
│   └── migrate_registry.py         # Re-register models + promote @production
├── src/
│   ├── config.py                   # ⭐ Cấu hình tập trung (MLflow URI, class names, API port, ...)
│   ├── pipeline/
│   │   └── pipeline.py             # ⭐ 5-step MLOps orchestrator
│   ├── servering/
│   │   ├── api.py                  #   FastAPI endpoints (predict, health, metrics, model-info)
│   │   ├── app.py                  #   Streamlit frontend (upload MRI, hiển thị kết quả)
│   │   └── services.py             # ⭐ ModelService (4-tier fallback) + PredictionService (LRU cache)
│   ├── utils/
│   │   ├── model.py                #   YOLO load/train/predict helpers
│   │   ├── yolo_wrapper.py         # ⭐ Custom MLflow PyFunc wrapper (PT + ONNX)
│   │   └── onnx_export.py          #   ONNX export utility (dynamic axes, simplify)
│   ├── evaluation/
│   │   └── evaluate.py             #   Model evaluation + MLflow metric logging
│   ├── deployment/
│   │   ├── deploy_model.py         #   Environment-based deployment
│   │   └── register_model.py       #   CLI for registry management
│   ├── monitoring/
│   │   └── monitor_model.py        #   Inference monitoring
│   └── training/
│       └── train.py                #   CLI training entry point
├── tests/                          # Test suite (unit, integration, e2e)
├── notebooks/                      # Jupyter experiments
├── .github/workflows/ci.yml        # GitHub Actions CI/CD
├── Dockerfile                      # Multi-stage Docker build (3 stages)
├── docker-compose.yml              # Backend + Frontend orchestration
├── pyproject.toml                  # Dependencies (uv/pip)
├── uv.lock                        # Lock file (reproducible builds)
├── MLproject                       # MLflow project entry points
└── README.md
```

---

## 🚀 Cài Đặt & Khởi Chạy

### Yêu cầu hệ thống

- Python 3.12+
- NVIDIA GPU + CUDA 11.8 (khuyến nghị) hoặc CPU
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Clone và cài đặt dependencies

```bash
git clone https://github.com/your-username/mlops_brain_turmo.git
cd mlops_brain_turmo

# Cài uv nếu chưa có
curl -LsSf https://astral.sh/uv/install.sh | sh          # Linux/macOS
# hoặc: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Tạo virtualenv và cài tất cả dependencies
uv sync
```

### 2. Đăng ký model vào MLflow (lần đầu)

```bash
# Đăng ký deployed_models/v1/ dưới dạng PyFunc wrapper + gán alias @production
uv run python scripts/migrate_registry.py
```

### 3. Khởi chạy services

```bash
# Terminal 1 — FastAPI backend
uv run python -m uvicorn src.servering.api:app --host 127.0.0.1 --port 8000

# Terminal 2 — Streamlit frontend
uv run python -m streamlit run src/servering/app.py

# Terminal 3 — MLflow UI (tùy chọn)
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### 4. Truy cập

| Service | URL |
|---------|-----|
| **Frontend (Streamlit)** | http://localhost:8501 |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **MLflow UI** | http://localhost:5000 |

---

## 🔄 Pipeline MLOps

Pipeline orchestrator (`src/pipeline/pipeline.py`) chạy 5 bước tuần tự:

```
Step 1: Train         →  Huấn luyện YOLOv11s với MLflow autolog
Step 2: Validate      →  Đánh giá metrics (mAP, precision, recall)
Step 3: Export ONNX   →  Xuất ONNX với dynamic axes + simplify
Step 4: Save locally  →  Lưu bản phiên bản deployed_models/vN/
Step 5: Register      →  Đăng ký MLflow Registry dưới dạng PyFunc → alias @staging
```

### Chạy pipeline

```bash
# Full pipeline (train + validate + export + save + register)
uv run python -m src.pipeline.pipeline --epochs 100

# Bỏ qua bước đăng ký (chỉ local)
uv run python -m src.pipeline.pipeline --epochs 50 --skip-register

# Tuỳ chỉnh cấu hình
uv run python -m src.pipeline.pipeline \
    --data configs/data.yaml \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640
```

### Promote model lên production

```bash
# Sau khi kiểm tra model staged, promote lên production
uv run python scripts/migrate_registry.py --promote --version 2
```

---

## 🎯 Model Serving

### YOLOWrapper — Custom PyFunc

`YOLOWrapper` kế thừa `mlflow.pyfunc.PythonModel`, cho phép:

1. **Load trực tiếp** từ MLflow Registry qua `mlflow.pyfunc.load_model()`
2. **Bundle cả hai** PyTorch và ONNX weights trong cùng một model version
3. **Lazy-load ONNX** — zero extra RAM cho đến khi người dùng bật ONNX switch
4. **Trả về native YOLO Results** — code downstream hoạt động giống nhau cho cả hai backends

### Chiến lược fallback 4 tầng

```
@production alias  →  Latest registry version  →  deployed_models/vN/  →  yolo11s.pt
     ✅ Primary           ✅ Rollback               ✅ Offline cache       ⚠️ Last resort
```

### Chuyển đổi backend

Người dùng chuyển đổi giữa **PyTorch** và **ONNX** trực tiếp từ Streamlit sidebar. Switch được xử lý tại inference time qua `params={"use_onnx": True}` — không cần restart server.

---

## 📡 API Reference

### Endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/` | Trạng thái API |
| `GET` | `/health` | Health check (GPU, model status, pyfunc flag) |
| `GET` | `/metrics` | Thống kê prediction (cache hit rate, avg latency) |
| `GET` | `/model-info` | Model source, type, ONNX availability |
| `POST` | `/predict` | Upload ảnh MRI → phát hiện khối u |
| `GET` | `/mlflow-dashboard` | Thông tin truy cập MLflow UI |

### POST /predict

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@brain_mri.jpg" \
  -F "use_onnx=false" \
  -F "confidence=0.5"
```

**Response:**

```json
{
  "predictions": [
    {
      "class": "glioma",
      "confidence": 0.92,
      "bbox": [120.5, 80.3, 350.2, 290.1]
    }
  ],
  "processing_time": 0.045,
  "model_type": "Registry/@production",
  "model_framework": "PyTorch",
  "request_id": "a1b2c3d4-..."
}
```

---

## ⚙️ Cấu Hình

Toàn bộ cấu hình tập trung trong `src/config.py`:

| Thiết lập | Mặc định | Biến môi trường |
|-----------|----------|-----------------|
| MLflow Tracking URI | `sqlite:///mlruns/mlflow.db` | `MLFLOW_TRACKING_URI` |
| API Host | `0.0.0.0` | `API_HOST` |
| API Port | `8000` | `API_PORT` |
| Registry Model Name | `brain_tumor_detector` | — |
| Prediction Cache Size | `50` | — |
| Default Confidence | `0.25` | — |
| Input Size | `640` | — |

---

## 📊 Hiệu Năng Mô Hình

### Validation Metrics

| Metric | Giá trị |
|--------|---------|
| **mAP@0.5** | 92.0% |
| **mAP@0.5:0.95** | 78.5% |
| **Precision** | 89.3% |
| **Recall** | 87.1% |

### Tốc độ inference

| Backend | Device | Latency |
|---------|--------|---------|
| PyTorch (.pt) | GPU (CUDA) | ~12ms |
| PyTorch (.pt) | CPU | ~50ms |
| ONNX (.onnx) | CPU | ~8ms |

---

## 🐳 Docker Deployment

### Build & chạy với Docker Compose

```bash
# Build cả 2 images (backend + frontend)
docker compose up --build -d

# Xem logs
docker compose logs -f

# Dừng services
docker compose down
```

### Kiến trúc Docker

| Stage | Base Image | Mục đích | Kích thước |
|-------|-----------|----------|-----------|
| `backend-builder` | `nvidia/cuda:11.8.0-runtime-ubuntu22.04` | Build dependencies (uv sync) | — (không giữ lại) |
| `backend` | `nvidia/cuda:11.8.0-runtime-ubuntu22.04` | Production GPU API server | ~4.8 GB |
| `frontend` | `python:3.12-slim` | Streamlit UI (CPU only) | ~300 MB |

### Volumes (docker-compose)

| Volume | Container Path | Mục đích |
|--------|---------------|----------|
| `./models` | `/app/models` | Training weights |
| `./mlruns` | `/app/mlruns` | MLflow tracking database + artifacts |
| `./deployed_models` | `/app/deployed_models` | Versioned model cache |

---

## 💻 Phát Triển & Kiểm Thử

### Chạy tests

```bash
# Tất cả tests
uv run pytest tests/ -v

# Chỉ unit tests
uv run pytest tests/unit/ -v

# Với coverage report
uv run pytest tests/ --cov=src
```

### Training model mới

```bash
# Full pipeline (5 bước)
uv run python -m src.pipeline.pipeline --epochs 100

# Chỉ training (standalone)
uv run python src/training/train.py --epochs 50
```

### MLflow tracking

```bash
# Khởi chạy MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Liệt kê các registered models
uv run python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient('sqlite:///mlruns/mlflow.db')
for rm in client.search_registered_models():
    versions = client.search_model_versions(f\"name='{rm.name}'\")
    print(f'{rm.name}: {[v.version for v in versions]}')
"
```

### Quy ước code

- **Config**: Tất cả constants trong `src/config.py` — không hardcode URIs hay paths
- **Model loading**: Luôn qua `ModelService.get_model()` — không load trực tiếp
- **MLflow APIs**: Sử dụng **aliases** (`@production`, `@staging`) — không dùng deprecated stages
- **Registration**: Sử dụng `mlflow.pyfunc.log_model()` với `YOLOWrapper` — không dùng `log_artifacts()`

---

## 🛠 Công Nghệ Sử Dụng

| Hạng mục | Công nghệ |
|----------|-----------|
| **Object Detection** | Ultralytics YOLOv11s |
| **Deep Learning** | PyTorch 2.7 (CUDA 11.8) |
| **Model Optimization** | ONNX / ONNX Runtime 1.23 |
| **Experiment Tracking** | MLflow 3.6 |
| **Model Registry** | MLflow Model Registry (Custom PyFunc wrapper) |
| **Backend API** | FastAPI 0.121 + Uvicorn |
| **Frontend UI** | Streamlit 1.51 |
| **Data Versioning** | DVC (Google Drive remote) |
| **Package Manager** | uv (Astral) |
| **Containerization** | Docker + Docker Compose (NVIDIA GPU support) |
| **CI/CD** | GitHub Actions |
| **Python** | 3.12+ |

---

## 📄 Giấy Phép

Dự án được phân phối theo giấy phép MIT — xem [LICENSE](LICENSE) để biết chi tiết.

---

<p align="center">
  <b>Built with ❤️ as an MLOps portfolio project</b>
</p>
