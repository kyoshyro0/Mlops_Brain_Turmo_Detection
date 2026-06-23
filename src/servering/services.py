"""
Core business logic for model loading, inference, and caching.

ModelService  – loads the model once with a multi-tier fallback strategy
               (Registry @production → older Registry versions → local
               deployed_models/vN/ → default yolo11s.pt).
PredictionService – runs inference, manages a true-LRU result cache,
                    and logs metrics to MLflow asynchronously to keep
                    response latency low.
"""

import os
import time
import hashlib
import uuid
import threading
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import UploadFile, HTTPException

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL,
    MLFLOW_EXPERIMENT_INFERENCE,
    DEPLOYED_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_CONFIDENCE,
    PREDICTION_CACHE_SIZE,
)
from src.utils.model import load_model, predict as yolo_predict


# ═══════════════════════════════════════════════════════════════════════════════
# ModelService
# ═══════════════════════════════════════════════════════════════════════════════

class ModelService:
    """Manages model loading with registry-first fallback strategy.

    Load order:
        0. MLflow Registry — ``@production`` alias  (authoritative)
        1. MLflow Registry — latest version          (fallback)
        2. ``deployed_models/vN/``                   (local validated cache)
        3. Default ``yolo11s.pt``                    (last resort)
    """

    def __init__(self, project_root: str):
        self.project_root = project_root
        self._model = None
        self._model_source: str = ""
        self._model_type: str = ""
        self._is_pyfunc: bool = False

    # ── public API ─────────────────────────────────────────────────────────

    def get_model(self):
        """Return the cached model or load it for the first time.

        Returns:
            (model, source_label, type_label)
        """
        if self._model is not None:
            return self._model, self._model_source, self._model_type

        # 0. MLflow Registry @production alias
        loaded = self._try_registry_alias("production")
        if loaded:
            return loaded

        # 1. MLflow Registry — latest version (fallback)
        loaded = self._try_registry_latest()
        if loaded:
            return loaded

        # 2. Local validated cache — deployed_models/vN/
        loaded = self._try_local_versions()
        if loaded:
            return loaded

        # 3. Default pretrained model
        print(f"⚠️  No validated model found, loading default {DEFAULT_MODEL_PATH}")
        self._model = load_model(DEFAULT_MODEL_PATH)
        self._model_source = "Default"
        self._model_type = "PyTorch"
        self._is_pyfunc = False
        return self._model, self._model_source, self._model_type

    @property
    def is_pyfunc(self) -> bool:
        """True when the active model is a PyFunc wrapper (supports params)."""
        return self._is_pyfunc

    # ── private helpers ────────────────────────────────────────────────────

    def _try_registry_alias(self, alias: str):
        """Try loading from a specific Registry alias.

        Loads via ``mlflow.pyfunc.load_model`` then extracts the underlying
        ``YOLOWrapper`` instance to avoid PyFuncModel.predict() overhead
        (which converts numpy images to DataFrames — extremely slow).
        """
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MLFLOW_REGISTRY_MODEL}@{alias}"
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            # Extract the underlying YOLOWrapper for direct inference
            self._model = pyfunc_model._model_impl.python_model
            self._model_source = f"Registry/@{alias}"
            self._model_type = "PyFunc"
            self._is_pyfunc = True
            print(f"✅ Loaded model from MLflow Registry (@{alias})")
            return self._model, self._model_source, self._model_type
        except Exception as e:
            print(f"⚠️  Registry @{alias} load skipped: {e}")
            return None

    def _try_registry_latest(self):
        """Try loading the latest version from the Registry (any alias)."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            # Get all versions, sorted newest first
            versions = client.search_model_versions(
                f"name='{MLFLOW_REGISTRY_MODEL}'",
                order_by=["version_number DESC"],
                max_results=5,
            )
            for mv in versions:
                try:
                    model_uri = f"models:/{MLFLOW_REGISTRY_MODEL}/{mv.version}"
                    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
                    self._model = pyfunc_model._model_impl.python_model
                    self._model_source = f"Registry/v{mv.version}"
                    self._model_type = "PyFunc"
                    self._is_pyfunc = True
                    print(f"✅ Loaded model from MLflow Registry (version {mv.version})")
                    return self._model, self._model_source, self._model_type
                except Exception:
                    continue
        except Exception as e:
            print(f"⚠️  Registry latest-version scan skipped: {e}")
        return None

    def _try_local_versions(self):
        """Load from the latest deployed_models/vN/ folder."""
        latest_dir = self._find_latest_version_dir()
        if not latest_dir:
            return None

        pt_path = os.path.join(latest_dir, "best.pt")
        ver_name = os.path.basename(latest_dir)

        if os.path.exists(pt_path):
            print(f"✅ Loading validated model from {ver_name}")
            self._model = load_model(pt_path)
            self._model_source = f"Local/{ver_name}"
            self._model_type = "PyTorch"
            self._is_pyfunc = False
            return self._model, self._model_source, self._model_type
        return None

    def _find_latest_version_dir(self) -> str:
        """Find the latest deployed_models/vN/ folder."""
        base = os.path.join(self.project_root, DEPLOYED_DIR)
        if not os.path.exists(base):
            return None
        versions = [
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and d.startswith("v") and d[1:].isdigit()
        ]
        if not versions:
            return None
        latest = max(versions, key=lambda d: int(d[1:]))
        return os.path.join(base, latest)


# ═══════════════════════════════════════════════════════════════════════════════
# PredictionService
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionService:
    """Handles the full prediction pipeline: decode → cache → infer → log."""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self._cache: OrderedDict = OrderedDict()
        self.MAX_CACHE_SIZE = PREDICTION_CACHE_SIZE

        # Runtime counters for /metrics
        self._total_predictions = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_inference_time = 0.0

    # ── image processing ───────────────────────────────────────────────────

    async def process_image(self, file: UploadFile):
        """Decode an uploaded image into a NumPy array (RGB)."""
        if file.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG and PNG are supported.",
            )

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), contents

    # ── caching (true LRU) ─────────────────────────────────────────────────

    @staticmethod
    def _cache_key(file_content: bytes, confidence: float, use_onnx: bool) -> str:
        file_hash = hashlib.md5(file_content).hexdigest()
        return f"{file_hash}_{confidence}_{use_onnx}"

    def _cache_get(self, key: str):
        """Get from cache and move to end (most recently used)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: dict):
        """Insert with LRU eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self.MAX_CACHE_SIZE:
            self._cache.popitem(last=False)

    # ── result formatting ──────────────────────────────────────────────────

    @staticmethod
    def format_predictions(results, confidence_threshold: float) -> list:
        """Convert YOLO Results to a JSON-serialisable list of dicts."""
        predictions = []
        if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= confidence_threshold:
                    predictions.append({
                        "class": results[0].names[int(box.cls[0])],
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist(),
                    })
        return predictions

    # ── main prediction pipeline ───────────────────────────────────────────

    async def predict(
        self,
        file: UploadFile,
        use_onnx: bool,
        confidence: float,
        model,
        model_source: str,
        model_type: str,
    ) -> dict:
        """Full inference pipeline: decode → cache → infer → async-log → respond."""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # 1. Decode image
        image_np, file_content = await self.process_image(file)

        # 2. Cache lookup
        cache_key = self._cache_key(file_content, confidence, use_onnx)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            cached_copy = dict(cached)
            cached_copy["request_id"] = request_id
            cached_copy["processing_time"] = time.time() - start_time
            cached_copy["message"] = "Prediction retrieved from cache"
            return cached_copy

        self._cache_misses += 1

        # 3. Inference — call YOLOWrapper directly (bypasses PyFunc overhead)
        if self.model_service.is_pyfunc:
            results = model.predict(
                None,  # context (unused at inference time)
                image_np,
                params={"use_onnx": use_onnx, "confidence": confidence},
            )
        else:
            results = yolo_predict(model, image_np, conf_thres=confidence)

        # 4. Format
        predictions = self.format_predictions(results, confidence)

        processing_time = time.time() - start_time
        self._total_predictions += 1
        self._total_inference_time += processing_time

        # 5. Async MLflow logging (fire-and-forget — does not block response)
        framework = "ONNX" if use_onnx and self.model_service.is_pyfunc else model_type
        threading.Thread(
            target=self._log_to_mlflow,
            args=(request_id, model_source, framework, file.filename,
                  processing_time, predictions),
            daemon=True,
        ).start()

        # 6. Build response & cache
        response = {
            "predictions": predictions,
            "message": "Prediction completed successfully",
            "processing_time": processing_time,
            "model_type": model_source,
            "model_framework": framework,
            "request_id": request_id,
        }
        self._cache_put(cache_key, response)
        return response

    # ── async MLflow logging ───────────────────────────────────────────────

    def _log_to_mlflow(
        self,
        request_id: str,
        model_type: str,
        framework: str,
        filename: str,
        processing_time: float,
        predictions: list,
    ):
        """Log inference metrics to MLflow in a background thread."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_INFERENCE)

            with mlflow.start_run(
                run_name=f"inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            ):
                mlflow.log_params({
                    "request_id": request_id,
                    "model_type": model_type,
                    "model_framework": framework,
                    "filename": filename or "unknown",
                })
                mlflow.log_metrics({
                    "processing_time": processing_time,
                    "num_detections": len(predictions),
                })
                if predictions:
                    confidences = [p["confidence"] for p in predictions]
                    mlflow.log_metrics({
                        "avg_confidence": sum(confidences) / len(confidences),
                        "max_confidence": max(confidences),
                    })
        except Exception as e:
            print(f"MLflow logging failed (non-blocking): {e}")

    # ── metrics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return runtime statistics for the /metrics endpoint."""
        total = self._total_predictions
        total_requests = self._cache_hits + self._cache_misses
        return {
            "total_predictions": total,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(self._cache_hits / total_requests, 4) if total_requests > 0 else 0.0,
            "avg_inference_time": round(self._total_inference_time / total, 4) if total > 0 else None,
        }
