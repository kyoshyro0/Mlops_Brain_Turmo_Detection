"""
Model registry management for MLflow.

This module provides functions to register, version, and manage models
in MLflow Model Registry with support for multiple model formats.
"""

import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional


def register_model(
    run_id: str,
    model_name: str,
    stage: str,
    model_path: Optional[str] = None,
    description: Optional[str] = None,
    model_format: Optional[str] = None
):
    """
    Register model to MLflow Model Registry.
    
    Args:
        run_id: MLflow run ID containing the model
        model_name: Name to register the model under
        stage: Stage for the model (Production, Staging, Archived, None)
        model_path: Path to model file (optional, for logging as artifact)
        description: Description for this model version
        model_format: Model format to register (onnx, pt, pth)
        
    Returns:
        ModelVersion object from MLflow
        
    Raises:
        Exception: If model registration fails
    """
    # Setup MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # Store model format in environment for later use
    if model_format:
        os.environ["MODEL_FORMAT"] = model_format
    
    # Log model file as artifact if provided
    if model_path and os.path.exists(model_path):
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(model_path, "model")
    
    # Register model
    try:
        # Check if model already exists
        try:
            model_details = client.get_registered_model(model_name)
            print(f"Model {model_name} already exists in registry")
        except:
            # Create model if it doesn't exist
            client.create_registered_model(model_name)
            print(f"Created new model {model_name} in registry")
        
        # Default model URI
        model_uri = f"runs:/{run_id}/model"
        
        # Check for specific model format
        model_format = os.environ.get("MODEL_FORMAT", "").lower()
        if model_format in ["onnx", "pt", "pth"]:
            # Search for model file with desired format
            artifacts = client.list_artifacts(run_id, "model")
            format_exists = False
            
            for artifact in artifacts:
                if not artifact.is_dir and os.path.splitext(artifact.path)[1].lower() == f".{model_format}":
                    # Found model with desired format
                    model_uri = f"runs:/{run_id}/{artifact.path}"
                    format_exists = True
                    print(f"Registering {model_format} model: {artifact.path}")
                    break
            
            if not format_exists:
                print(f"Warning: Model with format {model_format} not found, using entire model directory")
        
        # Register model with determined URI
        model_version = mlflow.register_model(
            model_uri,
            model_name
        )
        
        print(f"Registered model {model_name} version {model_version.version}")
        
        # Add description if provided
        if description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
            print(f"Added description for version {model_version.version}")
        
        # Transition to stage if specified
        if stage:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            print(f"Model {model_name} version {model_version.version} is now in {stage} stage")
        
        return model_version
    
    except Exception as e:
        print(f"Error registering model: {str(e)}")
        raise


def list_model_versions(model_name: str = "brain_tumor_detector"):
    """
    List all versions of a registered model.
    
    Args:
        model_name: Name of model in registry
        
    Raises:
        Exception: If model not found or listing fails
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    try:
        # Get model details
        model_details = client.get_registered_model(model_name)
        print(f"Model: {model_name}")
        
        # List versions
        print("\nVersion List:")
        for version in model_details.latest_versions:
            print(f"Version: {version.version}")
            print(f"  Stage: {version.current_stage}")
            print(f"  Created: {version.creation_timestamp}")
            print(f"  Description: {version.description or 'No description'}")
            print(f"  Run ID: {version.run_id}")
            print()
            
    except Exception as e:
        print(f"Error retrieving model info: {str(e)}")
        raise


def transition_model_stage(model_name: str, version: str, stage: str):
    """
    Transition model version to a different stage.
    
    Args:
        model_name: Name of model in registry
        version: Model version to transition
        stage: Target stage (Production, Staging, Archived, None)
        
    Raises:
        Exception: If stage transition fails
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} version {version} transitioned to {stage}")
    except Exception as e:
        print(f"Error transitioning stage: {str(e)}")
        raise


def get_model_details(model_name: str = "brain_tumor_detector", version: Optional[str] = None):
    """
    Get detailed information about model and its format.
    
    Args:
        model_name: Name of model in registry
        version: Specific version (if None, gets all versions)
        
    Raises:
        Exception: If model details retrieval fails
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    try:
        # Get model details
        model_details = client.get_registered_model(model_name)
        print(f"Model: {model_name}")
        
        versions_to_check = []
        if version:
            # Get specific version
            model_version = client.get_model_version(model_name, version)
            versions_to_check = [model_version]
        else:
            # Get all versions
            versions_to_check = model_details.latest_versions
        
        print("\nDetailed Information:")
        for ver in versions_to_check:
            print(f"Version: {ver.version}")
            print(f"  Stage: {ver.current_stage}")
            print(f"  Created: {ver.creation_timestamp}")
            print(f"  Description: {ver.description or 'No description'}")
            print(f"  Run ID: {ver.run_id}")
            
            # Get artifact information to determine model format
            run = mlflow.get_run(ver.run_id)
            artifacts_uri = run.info.artifact_uri
            
            # Check model directory
            model_artifacts = client.list_artifacts(ver.run_id, "model")
            
            if model_artifacts:
                print("  Model formats:")
                for artifact in model_artifacts:
                    if artifact.is_dir:
                        print(f"    - Directory: {artifact.path}")
                    else:
                        file_ext = os.path.splitext(artifact.path)[1]
                        if file_ext in ['.pt', '.pth']:
                            print(f"    - PyTorch model: {artifact.path}")
                        elif file_ext == '.onnx':
                            print(f"    - ONNX model: {artifact.path}")
                        elif file_ext in ['.h5', '.keras']:
                            print(f"    - Keras/TensorFlow model: {artifact.path}")
                        else:
                            print(f"    - File: {artifact.path}")
            
            # Get deployment environment from tags
            tags = run.data.tags
            if 'deploy_env' in tags:
                print(f"  Deployment environment: {tags['deploy_env']}")
            else:
                print(f"  Deployment environment: Not specified")
            
            print()
            
    except Exception as e:
        print(f"Error retrieving model info: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage models in MLflow Model Registry")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Register command parser
    register_parser = subparsers.add_parser("register", help="Register new model")
    register_parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID")
    register_parser.add_argument("--model-name", type=str, help="Name to register model")
    register_parser.add_argument("--model-path", type=str, help="Path to model file")
    register_parser.add_argument("--description", type=str, help="Description for model version")
    register_parser.add_argument("--stage", type=str, default="Staging",
                                choices=["Production", "Staging", "Archived", "None"], 
                                help="Stage for model")
    register_parser.add_argument("--model-format", type=str, choices=["onnx", "pt", "pth"], 
                                help="Model format to register (onnx, pt, pth)")
    
    # List command parser
    list_parser = subparsers.add_parser("list", help="List model versions")
    list_parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Model name in registry")
    
    # Transition command parser
    transition_parser = subparsers.add_parser("transition", help="Transition model stage")
    transition_parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Model name in registry")
    transition_parser.add_argument("--version", type=str, required=True, help="Model version")
    transition_parser.add_argument("--stage", type=str, required=True, 
                                  choices=["Production", "Staging", "Archived", "None"], 
                                  help="Target stage")
    
    # Details command parser
    details_parser = subparsers.add_parser("details", help="Get detailed model information")
    details_parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Model name in registry")
    details_parser.add_argument("--version", type=str, help="Specific model version (if not provided, shows all)")

    # Promote command parser — shortcut for transition → Production
    promote_parser = subparsers.add_parser("promote", help="Promote a Staging model version to Production")
    promote_parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Model name in registry")
    promote_parser.add_argument("--version", type=str, required=True, help="Model version to promote")

    args = parser.parse_args()

    if args.command == "register":
        register_model(args.run_id, args.model_name, args.stage, args.model_path, args.description, args.model_format)
    elif args.command == "list":
        list_model_versions(args.model_name)
    elif args.command == "transition":
        transition_model_stage(args.model_name, args.version, args.stage)
    elif args.command == "promote":
        transition_model_stage(args.model_name, args.version, "Production")
        print(f"✅ '{args.model_name}' version {args.version} is now in Production.")
        print("   Restart the API to load the new model.")
    elif args.command == "details":
        get_model_details(args.model_name, args.version)
    else:
        parser.print_help()