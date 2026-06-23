"""
FastAPI backend for brain tumor detection.

This module provides REST API endpoints for model inference, health checks,
and model management using YOLOv11 for brain tumor detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from datetime import datetime

# Define project root directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_INFERENCE,
    API_HOST,
    API_PORT,
)
from src.servering.services import ModelService, PredictionService

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for detecting brain tumors using YOLOv11s model",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Services (initialised, model loaded lazily in startup event) ───────────────

model_service = ModelService(PROJECT_ROOT)
prediction_service = PredictionService(model_service)

# These are populated in the startup event — not at import time.
model = None
model_source = ""
model_type = ""


@app.on_event("startup")
async def startup():
    """Load model once at server start — never at import time."""
    global model, model_source, model_type

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_INFERENCE)

    model, model_source, model_type = model_service.get_model()


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Root endpoint returning API status."""
    return {
        "message": "Brain Tumor Detection API is running",
        "version": "2.0.0",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    import torch

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_source": model_source,
        "model_type": model_type,
        "is_pyfunc": model_service.is_pyfunc,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/metrics")
async def get_metrics():
    """Get API metrics including prediction statistics."""
    return prediction_service.get_stats()


@app.get("/model-info")
async def model_info():
    """Get information about the currently loaded model.

    Returns model source, type, and whether ONNX switching is available.
    """
    onnx_available = False
    if model_service.is_pyfunc and hasattr(model, "onnx_available"):
        # YOLOWrapper extracted from PyFunc — check ONNX artifact
        onnx_available = model.onnx_available

    return {
        "model_source": model_source,
        "model_type": model_type,
        "is_pyfunc": model_service.is_pyfunc,
        "onnx_available": onnx_available,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_onnx: bool = False,
    confidence: float = 0.5,
):
    """Detect brain tumors in an uploaded MRI image.

    Args:
        file:       Uploaded image file (JPEG or PNG)
        use_onnx:   Use ONNX backend for inference (faster on CPU)
        confidence: Confidence threshold for detections (0.0–1.0)

    Returns:
        Predictions with bounding boxes, confidence scores, and metadata

    Raises:
        HTTPException: If file is invalid or prediction fails
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG are supported.",
        )

    try:
        result = await prediction_service.predict(
            file=file,
            use_onnx=use_onnx,
            confidence=confidence,
            model=model,
            model_source=model_source,
            model_type=model_type,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/mlflow-dashboard")
async def mlflow_dashboard_info():
    """Get MLflow UI access information."""
    return {
        "mlflow_url": "http://localhost:5000",
        "message": "Run: mlflow ui --backend-store-uri sqlite:///mlflow.db",
        "note": "MLflow UI must be started separately",
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.servering.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )