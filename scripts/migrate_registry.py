"""
Migration script: re-register existing model(s) using PyFunc wrapper.

Usage:
    # Re-register deployed_models/v1/ and set as production
    python scripts/migrate_registry.py

    # Promote an existing version to production
    python scripts/migrate_registry.py --promote --version 3
"""

import argparse
import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_MODEL,
    MLFLOW_EXPERIMENT_TRAINING,
    DEPLOYED_DIR,
)
from src.utils.yolo_wrapper import YOLOWrapper


def register_local_version(version_dir: str) -> str:
    """Register a local deployed_models/vN/ directory as a PyFunc model.

    Returns:
        The new MLflow model version number (string).
    """
    pt_path = os.path.join(version_dir, "best.pt").replace("\\", "/")
    onnx_path = os.path.join(version_dir, "best.onnx").replace("\\", "/")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"best.pt not found in {version_dir}")

    artifacts = {"best_pt": pt_path}
    if os.path.exists(onnx_path):
        artifacts["best_onnx"] = onnx_path

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_TRAINING)
    client = MlflowClient()

    # Ensure registered model exists
    try:
        client.get_registered_model(MLFLOW_REGISTRY_MODEL)
    except Exception:
        client.create_registered_model(
            MLFLOW_REGISTRY_MODEL,
            description="Brain tumor detection model (YOLOv11s, PT + ONNX)",
        )
        print(f"   Created registered model: {MLFLOW_REGISTRY_MODEL}")

    # Log model with PyFunc wrapper
    with mlflow.start_run(run_name=f"migration-{os.path.basename(version_dir)}") as run:
        mlflow.pyfunc.log_model(
            artifact_path="yolo_model",
            python_model=YOLOWrapper(),
            artifacts=artifacts,
        )
        run_id = run.info.run_id

    # Register
    model_uri = f"runs:/{run_id}/yolo_model"
    mv = mlflow.register_model(model_uri, MLFLOW_REGISTRY_MODEL)
    print(f"✅ Registered version {mv.version} from {version_dir}")

    return mv.version


def promote_to_production(version: str):
    """Set the 'production' alias on a specific model version."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    client.set_registered_model_alias(
        name=MLFLOW_REGISTRY_MODEL,
        alias="production",
        version=version,
    )
    print(f"✅ Version {version} promoted to @production")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate/promote models in MLflow Registry",
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="Promote a version to production (requires --version)",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Model version to promote",
    )
    parser.add_argument(
        "--version-dir", type=str, default=None,
        help="Local deployed_models/vN/ directory to register (default: auto-detect latest)",
    )

    args = parser.parse_args()

    if args.promote:
        if not args.version:
            print("Error: --promote requires --version")
            sys.exit(1)
        promote_to_production(args.version)
        return

    # Default: register + promote the latest local version
    if args.version_dir:
        version_dir = args.version_dir
    else:
        # Auto-detect latest deployed_models/vN/
        base = os.path.join(PROJECT_ROOT, DEPLOYED_DIR)
        if not os.path.exists(base):
            print(f"Error: {base} does not exist")
            sys.exit(1)
        versions = [
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and d.startswith("v") and d[1:].isdigit()
        ]
        if not versions:
            print(f"Error: No vN/ folders found in {base}")
            sys.exit(1)
        latest = max(versions, key=lambda d: int(d[1:]))
        version_dir = os.path.join(base, latest)

    print(f"📦 Registering {version_dir} ...")
    new_version = register_local_version(version_dir)

    print(f"🚀 Promoting version {new_version} to @production ...")
    promote_to_production(new_version)

    print(f"\n🎉 Migration complete! Model v{new_version} is now @production")
    print(f"   Verify: mlflow.pyfunc.load_model('models:/{MLFLOW_REGISTRY_MODEL}@production')")


if __name__ == "__main__":
    main()
