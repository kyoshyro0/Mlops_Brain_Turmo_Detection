# MLflow — Hướng Dẫn Sử Dụng

Tài liệu hướng dẫn sử dụng MLflow trong dự án Brain Tumor Detection, bao gồm Experiment Tracking, Model Registry, và PyFunc Wrapper.

---

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Cấu Hình](#cấu-hình)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [PyFunc Wrapper](#pyfunc-wrapper)
- [MLflow UI](#mlflow-ui)
- [CLI Commands](#cli-commands)

---

## Tổng Quan

Dự án sử dụng MLflow cho 3 mục đích chính:

| Chức năng | Mô tả |
|-----------|--------|
| **Experiment Tracking** | Log params, metrics, artifacts cho mọi workflow (training, evaluation, inference, deployment, monitoring) |
| **Model Registry** | Quản lý phiên bản model với aliases (`@production`, `@staging`) |
| **PyFunc Wrapper** | Custom `YOLOWrapper` cho phép load model trực tiếp qua `mlflow.pyfunc.load_model()` |

### Backend lưu trữ

- **Tracking URI**: `sqlite:///mlruns/mlflow.db` (SQLite database)
- **Artifact Store**: Thư mục `mlruns/` trên local filesystem
- Có thể override qua biến môi trường `MLFLOW_TRACKING_URI`

---

## Cấu Hình

Tất cả cấu hình MLflow nằm trong `src/config.py`:

```python
# Tracking URI
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")

# Model Registry
MLFLOW_REGISTRY_MODEL = "brain_tumor_detector"

# Experiment names
MLFLOW_EXPERIMENT_TRAINING   = "brain_tumor_detection"
MLFLOW_EXPERIMENT_INFERENCE  = "brain_tumor_detection_inference"
MLFLOW_EXPERIMENT_EVALUATION = "model_evaluation"
MLFLOW_EXPERIMENT_DEPLOYMENT = "model_deployment"
MLFLOW_EXPERIMENT_MONITORING = "model_monitoring"
```

> **Quy ước:** Luôn sử dụng `MLFLOW_TRACKING_URI` từ config, không hardcode URI trong code.

---

## Experiment Tracking

### Experiments trong dự án

| Experiment | Mô tả | Được gọi từ |
|------------|--------|-------------|
| `brain_tumor_detection` | Training runs | `src/utils/model.py` → `train_model()` |
| `model_evaluation` | Evaluation metrics | `src/evaluation/evaluate.py` |
| `brain_tumor_detection_inference` | Inference metrics (mỗi request) | `src/servering/services.py` → `PredictionService` |
| `model_deployment` | Deployment logs | `src/deployment/deploy_model.py` |
| `model_monitoring` | Monitoring results | `src/monitoring/monitor_model.py` |

### Training Experiment

Khi chạy pipeline hoặc training standalone, MLflow tự động log:

**Parameters:**
- `epochs`, `imgsz`, `batch_size`, `device`

**Metrics:**
- `mAP50`, `mAP50_95`, `precision`, `recall`

**Artifacts:**
- `weights/best.pt` — PyTorch checkpoint tốt nhất

```python
# Ví dụ: Training với MLflow autolog
mlflow.pytorch.autolog()
with mlflow.start_run() as run:
    results = model.train(data=data_yaml, epochs=100, ...)
    mlflow.log_metrics({
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        ...
    })
```

### Inference Experiment

Mỗi request `/predict` được log **bất đồng bộ** trong background thread:

**Parameters:** `request_id`, `model_type`, `model_framework`, `filename`

**Metrics:** `processing_time`, `num_detections`, `avg_confidence`, `max_confidence`

> Async logging đảm bảo response latency không bị ảnh hưởng. Exception được catch và print, không block response.

---

## Model Registry

### Tên model

Tất cả model versions được đăng ký dưới tên: **`brain_tumor_detector`**

### Aliases (thay thế deprecated Stages)

| Alias | Mô tả | Khi nào được gán |
|-------|--------|-------------------|
| `@production` | Model đang phục vụ production | Promote thủ công qua `migrate_registry.py` |
| `@staging` | Model mới chờ kiểm tra | Tự động gán khi pipeline register |

> **Quan trọng:** Dự án sử dụng **aliases** (`@production`, `@staging`), KHÔNG sử dụng deprecated stages (`Production`, `Staging`).

### Đăng ký model mới

Pipeline tự động đăng ký model sau bước 5:

```python
# Đăng ký PyFunc wrapper
mlflow.pyfunc.log_model(
    artifact_path="yolo_model",
    python_model=YOLOWrapper(),
    artifacts={
        "best_pt": "deployed_models/v1/best.pt",
        "best_onnx": "deployed_models/v1/best.onnx",
    },
)

# Register vào Registry
model_version = mlflow.register_model(
    f"runs:/{run_id}/yolo_model",
    "brain_tumor_detector"
)

# Gán alias staging
client.set_registered_model_alias(
    name="brain_tumor_detector",
    alias="staging",
    version=model_version.version,
)
```

### Promote lên production

```bash
uv run python scripts/migrate_registry.py --promote --version <N>
```

Script sẽ gán alias `@production` cho version chỉ định.

### Load model từ Registry

```python
import mlflow.pyfunc

# Load qua alias
model = mlflow.pyfunc.load_model("models:/brain_tumor_detector@production")

# Load qua version number
model = mlflow.pyfunc.load_model("models:/brain_tumor_detector/3")
```

---

## PyFunc Wrapper

### Tại sao cần YOLOWrapper?

MLflow không hỗ trợ native Ultralytics YOLO. `YOLOWrapper` giải quyết:

1. **Load trực tiếp** — `mlflow.pyfunc.load_model()` load cả PyTorch + ONNX weights
2. **Bundle đôi** — Một model version chứa cả `.pt` và `.onnx`
3. **Lazy ONNX** — ONNX chỉ load khi lần đầu được yêu cầu
4. **Native Results** — Trả về YOLO Results object, downstream code hoạt động giống nhau

### Cách hoạt động

```
mlflow.pyfunc.load_model()
    └── YOLOWrapper.load_context(context)
            ├── Load best.pt → self.pt_model (always)
            └── Store onnx path → self._onnx_path (lazy)

YOLOWrapper.predict(context, image, params)
    ├── params["use_onnx"] == True  → self.onnx_model(image)  (lazy load)
    └── params["use_onnx"] == False → self.pt_model(image)
```

### Tại sao extract YOLOWrapper thay vì gọi PyFuncModel.predict()?

```python
# ❌ KHÔNG LÀM THẾ NÀY — cực kỳ chậm
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
result = pyfunc_model.predict(image_array)  # Converts to DataFrame!

# ✅ LÀM THẾ NÀY — extract wrapper, gọi trực tiếp
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
wrapper = pyfunc_model._model_impl.python_model  # YOLOWrapper instance
result = wrapper.predict(None, image_array, params={"use_onnx": False})
```

`PyFuncModel.predict()` convert numpy array sang pandas DataFrame → rất chậm cho images. Extract `YOLOWrapper` cho phép gọi predict() trực tiếp.

---

## MLflow UI

### Khởi chạy

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Truy cập: http://localhost:5000

### Các tab chính

| Tab | Mô tả |
|-----|--------|
| **Experiments** | Danh sách experiments, compare runs, xem metrics/params |
| **Models** | Registry — xem các model versions, aliases, artifacts |

---

## CLI Commands

### Liệt kê registered models

```bash
uv run python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient('sqlite:///mlruns/mlflow.db')
for rm in client.search_registered_models():
    versions = client.search_model_versions(f\"name='{rm.name}'\")
    print(f'{rm.name}: {[v.version for v in versions]}')
"
```

### Đăng ký model mới từ local

```bash
# Register deployed_models/v1/ + promote @production
uv run python scripts/migrate_registry.py

# Register một thư mục cụ thể
uv run python scripts/migrate_registry.py --version-dir deployed_models/v2
```

### Promote model

```bash
uv run python scripts/migrate_registry.py --promote --version 3
```

### Quản lý model registry (CLI)

```bash
# Liệt kê versions
uv run python -m src.deployment.register_model list --model-name brain_tumor_detector

# Chi tiết model
uv run python -m src.deployment.register_model details --model-name brain_tumor_detector

# Promote lên Production (legacy stages)
uv run python -m src.deployment.register_model promote --version 2
```

### Deploy model

```bash
# Deploy từ Registry
uv run python -m src.deployment.deploy_model deploy \
    --model-name brain_tumor_detector \
    --version 2 \
    --env production

# Liệt kê deployments
uv run python -m src.deployment.deploy_model list --env production

# Xem deployment mới nhất
uv run python -m src.deployment.deploy_model latest --env production
```
