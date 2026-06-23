"""
Centralized configuration for the MLOps Brain Tumor Detection project.

All hardcoded values (MLflow URIs, model names, class labels, etc.) are
consolidated here so every module imports from a single source of truth.
Override any setting via environment variables where noted.
"""

import os

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"
)
MLFLOW_REGISTRY_MODEL = "brain_tumor_detector"

# Experiment names
MLFLOW_EXPERIMENT_TRAINING = "brain_tumor_detection"
MLFLOW_EXPERIMENT_INFERENCE = "brain_tumor_detection_inference"
MLFLOW_EXPERIMENT_EVALUATION = "model_evaluation"
MLFLOW_EXPERIMENT_DEPLOYMENT = "model_deployment"
MLFLOW_EXPERIMENT_MONITORING = "model_monitoring"

# ─── Model ────────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
DEFAULT_MODEL_PATH = "yolo11s.pt"
DEPLOYED_DIR = "deployed_models"
DEFAULT_CONFIDENCE = 0.25
INPUT_SIZE = 640

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# ─── Cache ────────────────────────────────────────────────────────────────────
PREDICTION_CACHE_SIZE = 50
