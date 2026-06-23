"""
MLOps Pipeline Orchestrator.

Runs the complete workflow:
    Train → Validate → Export ONNX → Save locally (versioned) → Register to MLflow

Each run produces a new version folder under deployed_models/vN/ containing
both best.pt and best.onnx. The same pair is registered as a single version
in the MLflow Model Registry under stage "Staging".

Usage:
    python -m src.pipeline.pipeline --epochs 100
    python -m src.pipeline.pipeline --epochs 50 --batch-size 32 --skip-register
"""

import os
import shutil
import argparse
from datetime import datetime
from typing import Tuple, Optional

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL,
    DEPLOYED_DIR,
)
from src.utils.yolo_wrapper import YOLOWrapper


# ─── Helper: version management ──────────────────────────────────────────────


def _next_version_number(base_dir: str = DEPLOYED_DIR) -> int:
    """Determine the next version number by scanning existing vN/ folders."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and d.startswith("v") and d[1:].isdigit()
    ]
    if not existing:
        return 1
    return max(int(d[1:]) for d in existing) + 1


def save_version_locally(
    best_pt: str,
    best_onnx: str,
    base_dir: str = DEPLOYED_DIR,
) -> Tuple[int, str]:
    """Copy best.pt + best.onnx into deployed_models/vN/.

    Args:
        best_pt:   Absolute or relative path to best.pt
        best_onnx: Absolute or relative path to best.onnx
        base_dir:  Root folder for versioned deployments

    Returns:
        (version_number, version_directory_path)
    """
    version = _next_version_number(base_dir)
    version_dir = os.path.join(base_dir, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    shutil.copy2(best_pt, os.path.join(version_dir, "best.pt"))
    shutil.copy2(best_onnx, os.path.join(version_dir, "best.onnx"))

    print(f"✅ Saved version v{version} → {version_dir}")
    print(f"   best.pt  : {os.path.getsize(os.path.join(version_dir, 'best.pt')) / 1e6:.1f} MB")
    print(f"   best.onnx: {os.path.getsize(os.path.join(version_dir, 'best.onnx')) / 1e6:.1f} MB")

    return version, version_dir


def register_version(
    run_id: str,
    version_dir: str,
    model_name: str = MLFLOW_REGISTRY_MODEL,
    description: str = "",
) -> object:
    """Register a trained model as a PyFunc wrapper in MLflow Registry.

    Logs both .pt and .onnx weights as artifacts inside a YOLOWrapper,
    enabling direct loading via ``mlflow.pyfunc.load_model()``.
    New versions receive the ``staging`` alias; promote to ``production``
    via ``scripts/migrate_registry.py`` or the CLI.

    Args:
        run_id:       MLflow run ID from the training step
        version_dir:  Local directory with best.pt and best.onnx
        model_name:   Model name in MLflow Model Registry
        description:  Description for this version

    Returns:
        MLflow ModelVersion object
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            model_name,
            description="Brain tumor detection model (YOLOv11s, PT + ONNX)",
        )
        print(f"   Created registered model: {model_name}")

    # Log as PyFunc wrapper (enables direct pyfunc.load_model)
    artifacts = {
        "best_pt": os.path.join(version_dir, "best.pt").replace("\\", "/"),
        "best_onnx": os.path.join(version_dir, "best.onnx").replace("\\", "/"),
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            artifact_path="yolo_model",
            python_model=YOLOWrapper(),
            artifacts=artifacts,
        )

    # Register the PyFunc model
    model_uri = f"runs:/{run_id}/yolo_model"
    model_version = mlflow.register_model(model_uri, model_name)

    # Add description
    if description:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description,
        )

    # New versions land with "staging" alias
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=model_version.version,
    )

    print(f"✅ Registered '{model_name}' version {model_version.version} → staging")
    print(f"   Promote: python scripts/migrate_registry.py --promote --version {model_version.version}")

    return model_version


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    data_yaml: str = "configs/data.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    model_name: str = "brain_tumor_detector",
    skip_register: bool = False,
):
    """Run the full MLOps pipeline.

    Steps:
        1. Train       → best.pt + MLflow run
        2. Validate    → evaluation metrics
        3. Export ONNX  → best.onnx
        4. Save locally → deployed_models/vN/
        5. Register    → MLflow Model Registry (Staging)

    Args:
        data_yaml:     Path to data configuration YAML
        epochs:        Number of training epochs
        batch_size:    Training batch size
        imgsz:         Input image size
        model_name:    Name in MLflow Model Registry
        skip_register: If True, skip step 5 (register)
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_pt_path = os.path.join("models", "train", "weights", "best.pt")

    print("\n" + "=" * 70)
    print(f"🚀 MLOps Pipeline Started at {timestamp}")
    print("=" * 70)

    # ── Step 1: Train ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("📊 Step 1/5: Training Model")
    print("─" * 70)

    from src.utils.model import train_model
    results, run_id = train_model(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
    )

    if not os.path.exists(best_pt_path):
        raise FileNotFoundError(f"Training did not produce {best_pt_path}")

    print(f"✅ Training complete. Run ID: {run_id}")

    # ── Step 2: Validate ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🔍 Step 2/5: Validating Model")
    print("─" * 70)

    from src.evaluation.evaluate import evaluate_model
    val_metrics = evaluate_model(
        model_path=best_pt_path,
        data_yaml=data_yaml,
        img_size=imgsz,
        batch_size=batch_size,
        log_mlflow=True,
    )

    mAP50 = val_metrics.get("metrics/mAP50(B)", 0)
    mAP50_95 = val_metrics.get("metrics/mAP50-95(B)", 0)
    print(f"✅ Validation complete. mAP@0.5: {mAP50:.4f} | mAP@0.5:0.95: {mAP50_95:.4f}")

    # ── Step 3: Export ONNX ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("⚙️  Step 3/5: Exporting to ONNX")
    print("─" * 70)

    from src.utils.onnx_export import export_to_onnx
    best_onnx_path = export_to_onnx(
        model_path=best_pt_path,
        dynamic=True,
        simplify=True,
    )

    if not os.path.exists(best_onnx_path):
        raise FileNotFoundError(f"ONNX export did not produce {best_onnx_path}")

    print(f"✅ ONNX export complete: {best_onnx_path}")

    # ── Step 4: Save locally (versioned) ──────────────────────────────────
    print("\n" + "─" * 70)
    print("💾 Step 4/5: Saving Versioned Files Locally")
    print("─" * 70)

    version, version_dir = save_version_locally(best_pt_path, best_onnx_path)

    # ── Step 5: Register to MLflow ────────────────────────────────────────
    model_version = None
    if not skip_register:
        print("\n" + "─" * 70)
        print("📦 Step 5/5: Registering to MLflow Model Registry")
        print("─" * 70)

        description = (
            f"v{version} | epochs={epochs}, imgsz={imgsz}, batch={batch_size} | "
            f"mAP@0.5={mAP50:.4f}, mAP@0.5:0.95={mAP50_95:.4f} | "
            f"run_id={run_id}"
        )

        model_version = register_version(
            run_id=run_id,
            version_dir=version_dir,
            model_name=model_name,
            description=description,
        )
    else:
        print("\n⏭️  Step 5/5: Registration skipped (--skip-register)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🎉 Pipeline Completed Successfully!")
    print("=" * 70)
    print(f"   MLflow Run ID  : {run_id}")
    print(f"   Local Version  : deployed_models/v{version}/")
    print(f"   Files          : best.pt + best.onnx")
    print(f"   mAP@0.5        : {mAP50:.4f}")
    print(f"   mAP@0.5:0.95   : {mAP50_95:.4f}")
    if model_version:
        print(f"   Registry       : {model_name} v{model_version.version} (Staging)")
        print(f"   Promote        : python -m src.deployment.register_model promote --version {model_version.version}")
    print("=" * 70)

    return {
        "run_id": run_id,
        "local_version": version,
        "version_dir": version_dir,
        "model_version": model_version,
        "metrics": val_metrics,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full MLOps pipeline: Train → Validate → Export ONNX → Save → Register"
    )
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Registry model name")
    parser.add_argument("--skip-register", action="store_true", help="Skip MLflow registration")

    args = parser.parse_args()

    run_pipeline(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        model_name=args.model_name,
        skip_register=args.skip_register,
    )
