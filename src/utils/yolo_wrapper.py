"""
Custom MLflow PyFunc wrapper for YOLO models.

Allows loading a trained YOLO model directly from the MLflow Model Registry
via ``mlflow.pyfunc.load_model()`` — no manual artifact download required.

The wrapper bundles **both** PyTorch (.pt) and ONNX (.onnx) weights inside a
single registered model version.  The caller selects the inference backend at
prediction time through ``params={"use_onnx": True/False}``.

Memory strategy:
    * PyTorch model is always loaded on init (primary backend).
    * ONNX model is **lazy-loaded** on first request — zero extra RAM until
      the user actually toggles the ONNX checkbox on the frontend.
"""

import os
import mlflow.pyfunc


class YOLOWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper around Ultralytics YOLO for registry serving."""

    # ── lifecycle ──────────────────────────────────────────────────────────

    def load_context(self, context):
        """Called automatically by MLflow when the model is loaded.

        ``context.artifacts`` is a dict mapping logical names to local file
        paths that MLflow downloaded/resolved from the artifact store:
            {"best_pt": "/tmp/.../best.pt", "best_onnx": "/tmp/.../best.onnx"}
        """
        from ultralytics import YOLO

        # Always load PyTorch (primary)
        pt_path = context.artifacts["best_pt"]
        self.pt_model = YOLO(pt_path)
        print(f"✅ YOLOWrapper: loaded PyTorch model from {pt_path}")

        # Store ONNX path for lazy loading
        self._onnx_path = context.artifacts.get("best_onnx")
        self._onnx_model = None

    # ── lazy ONNX loader ──────────────────────────────────────────────────

    @property
    def onnx_model(self):
        """Lazy-load ONNX model on first access — saves RAM until needed."""
        if self._onnx_model is None and self._onnx_path and os.path.exists(self._onnx_path):
            from src.utils.model import load_model
            self._onnx_model = load_model(self._onnx_path)
            print(f"✅ YOLOWrapper: lazy-loaded ONNX model from {self._onnx_path}")
        return self._onnx_model

    @property
    def onnx_available(self) -> bool:
        """Check if ONNX weights exist (without loading them)."""
        return bool(self._onnx_path and os.path.exists(self._onnx_path))

    # ── inference ──────────────────────────────────────────────────────────

    def predict(self, context, model_input, params=None):
        """Run inference through the selected backend.

        Args:
            context:     MLflow context (unused at inference time).
            model_input: numpy array — a single image (H, W, C) in RGB.
            params:      Optional dict with runtime knobs:
                         - ``use_onnx``   (bool)  — switch to ONNX backend.
                         - ``confidence`` (float) — detection threshold.

        Returns:
            YOLO-native ``Results`` list — identical format regardless of
            backend, so downstream code (``format_predictions``) works
            unchanged.
        """
        params = params or {}
        use_onnx = params.get("use_onnx", False)
        confidence = params.get("confidence", 0.25)

        if use_onnx and self.onnx_model is not None:
            results = self.onnx_model(model_input, conf=confidence)
        else:
            results = self.pt_model(model_input, conf=confidence)

        return results
