"""
Model deployment utilities for MLflow Registry.

This module provides functions to deploy models from MLflow Model Registry
to different environments (production, staging) with versioning and configuration management.
"""

import os
import argparse
import json
import shutil
import datetime
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict


def deploy_model_from_registry(
    model_name: str,
    version: Optional[int] = None,
    stage: Optional[str] = None,
    deploy_env: str = "Staging",
    config: Optional[Dict] = None
):
    """
    Deploy model from MLflow Registry to specified environment.
    
    Args:
        model_name: Name of model in registry
        version: Specific version number (if not provided, uses stage)
        stage: Model stage (Production, Staging, None)
        deploy_env: Deployment environment name
        config: Additional deployment configuration
        
    Returns:
        Dictionary with deployment information
        
    Raises:
        ValueError: If model not found in registry
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # Get model from registry
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
        model_version = client.get_model_version(model_name, version)
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
        # Get model version from stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"Model {model_name} not found in {stage} stage")
        model_version = versions[0]
    else:
        # Get latest Production version
        model_uri = f"models:/{model_name}/Production"
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise ValueError(f"Model {model_name} not found in Production stage")
        model_version = versions[0]
    
    print(f"Loading model from {model_uri}...")
    
    # Create deployment directory with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = model_version.run_id
    
    # Create deployment directory
    deploy_base_dir = os.path.join("deployed_models", deploy_env)
    deploy_dir = os.path.join(deploy_base_dir, f"{model_name}_{timestamp}")
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Download model to deployment directory
    print(f"Downloading model to: {deploy_dir}...")
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=deploy_dir)
    
    # Find model files in downloaded artifacts
    model_files = []
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(('.pt', '.pth', '.h5', '.keras', '.onnx', '.pkl')):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        raise ValueError(f"No model files found in downloaded artifacts")
    
    # Use first model file found
    model_path = model_files[0]
    model_filename = os.path.basename(model_path)
    model_format = os.path.splitext(model_filename)[1][1:]  # Get file extension without dot
    
    print(f"Found model file: {model_path}")
    
    # Path to deployed model
    deployed_model_path = model_path
    
    # Create deployment configuration file
    deployment_config = {
        "model_path": model_path,
        "deployed_at": datetime.datetime.now().isoformat(),
        "environment": deploy_env,
        "deployed_model_path": deployed_model_path,
        "model_name": model_name,
        "model_filename": model_filename,
        "model_format": model_format,
        "timestamp": timestamp,
        "mlflow_model_name": model_version.name,
        "mlflow_model_version": model_version.version,
        "mlflow_model_stage": model_version.current_stage,
        "mlflow_run_id": run_id,
        "config": config or {}
    }
    
    config_path = os.path.join(deploy_dir, "deployment_config.json")
    with open(config_path, "w") as f:
        json.dump(deployment_config, f, indent=4)
    
    # Create .env file for application to know which model is deployed
    env_path = os.path.join(deploy_dir, ".env")
    with open(env_path, "w") as f:
        f.write(f"MODEL_PATH={deployed_model_path}\n")
        f.write(f"DEPLOYMENT_ENV={deploy_env}\n")
        f.write(f"MODEL_NAME={model_name}\n")
        f.write(f"MODEL_FILENAME={model_filename}\n")
        f.write(f"MODEL_FORMAT={model_format}\n")
        f.write(f"DEPLOYMENT_TIMESTAMP={timestamp}\n")
        f.write(f"MLFLOW_MODEL_NAME={model_version.name}\n")
        f.write(f"MLFLOW_MODEL_VERSION={model_version.version}\n")
        f.write(f"MLFLOW_MODEL_STAGE={model_version.current_stage}\n")
        f.write(f"MLFLOW_RUN_ID={run_id}\n")
        if config:
            for key, value in config.items():
                f.write(f"{key.upper()}={value}\n")
    
    # Create/update latest deployment marker file
    latest_config_path = os.path.join(deploy_base_dir, "latest_deployment.json")
    latest_info = {
        "latest_deployment_dir": deploy_dir,
        "latest_model_path": deployed_model_path,
        "timestamp": timestamp,
        "model_name": model_name,
        "model_format": model_format,
        "mlflow_model_name": model_version.name,
        "mlflow_model_version": model_version.version,
        "mlflow_model_stage": model_version.current_stage,
        "mlflow_run_id": run_id
    }
    
    with open(latest_config_path, "w") as f:
        json.dump(latest_info, f, indent=4)
    
    print(f"Model deployed at: {deploy_dir}")
    print(f"This is the latest deployment for {deploy_env} environment")
    
    # Log deployment to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("model_deployment")
    
    with mlflow.start_run(run_name=f"deploy-{model_name}-{deploy_env}-{timestamp}"):
        mlflow.log_param("deploy_env", deploy_env)
        mlflow.log_param("deployed_model_path", deployed_model_path)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_format", model_format)
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_param("mlflow_model_name", model_version.name)
        mlflow.log_param("mlflow_model_version", model_version.version)
        mlflow.log_param("mlflow_model_stage", model_version.current_stage)
        mlflow.log_param("source_run_id", run_id)
        
        if config:
            for key, value in config.items():
                mlflow.log_param(f"config_{key}", value)
        
        # Log deployment config as artifact
        mlflow.log_artifact(config_path)
        mlflow.log_artifact(env_path)
    
    return deployment_config


def list_deployments(deploy_env: str = "Staging"):
    """
    List all deployments in specified environment.
    
    Args:
        deploy_env: Environment name to list deployments from
    """
    deploy_base_dir = os.path.join("deployed_models", deploy_env)
    
    if not os.path.exists(deploy_base_dir):
        print(f"No deployments found in {deploy_env} environment")
        return
    
    deployments = []
    for item in os.listdir(deploy_base_dir):
        item_path = os.path.join(deploy_base_dir, item)
        if os.path.isdir(item_path):
            config_path = os.path.join(item_path, "deployment_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    deployment_info = json.load(f)
                    deployments.append(deployment_info)
    
    if not deployments:
        print(f"No deployments found in {deploy_env} environment")
        return
    
    # Sort by timestamp (newest first)
    deployments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    print(f"\nDeployments in {deploy_env} environment:")
    print("=" * 80)
    for dep in deployments:
        print(f"\nModel: {dep.get('model_name', 'Unknown')}")
        print(f"  Format: {dep.get('model_format', 'Unknown')}")
        print(f"  Deployed: {dep.get('deployed_at', 'Unknown')}")
        print(f"  MLflow Version: {dep.get('mlflow_model_version', 'Unknown')}")
        print(f"  Stage: {dep.get('mlflow_model_stage', 'Unknown')}")
        print(f"  Path: {dep.get('deployed_model_path', 'Unknown')}")


def get_latest_deployment(deploy_env: str = "Staging") -> Optional[Dict]:
    """
    Get information about latest deployment in environment.
    
    Args:
        deploy_env: Environment name
        
    Returns:
        Dictionary with latest deployment info, or None if no deployments
    """
    deploy_base_dir = os.path.join("deployed_models", deploy_env)
    latest_config_path = os.path.join(deploy_base_dir, "latest_deployment.json")
    
    if os.path.exists(latest_config_path):
        with open(latest_config_path, "r") as f:
            return json.load(f)
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy models from MLflow Registry")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model from registry")
    deploy_parser.add_argument("--model-name", type=str, required=True, help="Model name in registry")
    deploy_parser.add_argument("--version", type=int, help="Specific model version")
    deploy_parser.add_argument("--stage", type=str, choices=["Production", "Staging", "Archived"],
                              help="Model stage")
    deploy_parser.add_argument("--env", type=str, default="Staging",
                              help="Deployment environment")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List deployments")
    list_parser.add_argument("--env", type=str, default="Staging",
                            help="Environment to list")
    
    # Latest command
    latest_parser = subparsers.add_parser("latest", help="Get latest deployment info")
    latest_parser.add_argument("--env", type=str, default="Staging",
                              help="Environment")
    
    args = parser.parse_args()
    
    if args.command == "deploy":
        deploy_model_from_registry(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage,
            deploy_env=args.env
        )
    elif args.command == "list":
        list_deployments(args.env)
    elif args.command == "latest":
        latest = get_latest_deployment(args.env)
        if latest:
            print(json.dumps(latest, indent=4))
        else:
            print(f"No deployments found in {args.env} environment")
    else:
        parser.print_help()