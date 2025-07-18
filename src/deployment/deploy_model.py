import os
import argparse
import json
import shutil
import datetime
import mlflow
from mlflow.tracking import MlflowClient

def deploy_model_from_registry(model_name, version=None, stage=None, deploy_env="Staging", config=None):
    """
    Triển khai model từ MLflow Registry vào môi trường cụ thể
    
    Args:
        model_name (str): Tên model trong registry
        version (int): Phiên bản cụ thể (nếu không cung cấp, sẽ sử dụng stage)
        stage (str): Giai đoạn của model (Production, Staging, None)
        deploy_env (str): Môi trường triển khai
        config (dict): Cấu hình triển khai bổ sung
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # Lấy model từ registry
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
        model_version = client.get_model_version(model_name, version)
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
        # Lấy phiên bản model từ stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"Không tìm thấy model {model_name} ở giai đoạn {stage}")
        model_version = versions[0]
    else:
        # Lấy phiên bản mới nhất trong Production
        model_uri = f"models:/{model_name}/Production"
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise ValueError(f"Không tìm thấy model {model_name} ở giai đoạn Production")
        model_version = versions[0]
    
    print(f"Đang tải model từ {model_uri}...")
    
    # Tạo timestamp cho thư mục triển khai
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = model_version.run_id
    
    # Tạo thư mục triển khai với timestamp, tên model và định dạng
    deploy_base_dir = os.path.join("deployed_models", deploy_env)
    deploy_dir = os.path.join(deploy_base_dir, f"{model_name}_{timestamp}")
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Tải model vào thư mục triển khai
    print(f"Đang tải model vào thư mục triển khai: {deploy_dir}...")
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=deploy_dir)
    
    # Tìm file model trong thư mục đã tải
    model_files = []
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(('.pt', '.pth', '.h5', '.keras', '.onnx', '.pkl')):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        raise ValueError(f"Không tìm thấy file model trong artifacts đã tải xuống")
    
    # Sử dụng file model đầu tiên tìm thấy
    model_path = model_files[0]
    model_filename = os.path.basename(model_path)
    model_name_from_file = os.path.splitext(model_filename)[0]  # Lấy tên model không có phần mở rộng
    model_format = os.path.splitext(model_filename)[1][1:]  # Lấy định dạng file (bỏ dấu chấm)
    
    print(f"Đã tìm thấy file model: {model_path}")
    
    # Đường dẫn đến model đã triển khai
    deployed_model_path = model_path
    shutil.copy2(model_path, deployed_model_path)
    
    # Tạo file cấu hình triển khai
    deployment_config = {
        "model_path": model_path,
        "deployed_at": datetime.datetime.now().isoformat(),
        "environment": deploy_env,
        "deployed_model_path": deployed_model_path,
        "model_name": model_name,
        "model_filename": model_filename,
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
    
    # Tạo file .env để ứng dụng biết model nào đang được sử dụng
    env_path = os.path.join(deploy_dir, ".env")
    with open(env_path, "w") as f:
        f.write(f"MODEL_PATH={deployed_model_path}\n")
        f.write(f"DEPLOYMENT_ENV={deploy_env}\n")
        f.write(f"MODEL_NAME={model_name}\n")
        f.write(f"MODEL_FILENAME={model_filename}\n")
        f.write(f"DEPLOYMENT_TIMESTAMP={timestamp}\n")
        f.write(f"MLFLOW_MODEL_NAME={model_version.name}\n")
        f.write(f"MLFLOW_MODEL_VERSION={model_version.version}\n")
        f.write(f"MLFLOW_MODEL_STAGE={model_version.current_stage}\n")
        f.write(f"MLFLOW_RUN_ID={run_id}\n")
        if config:
            for key, value in config.items():
                f.write(f"{key.upper()}={value}\n")
    
    # Tạo/cập nhật file chỉ định triển khai mới nhất
    latest_config_path = os.path.join(deploy_base_dir, "latest_deployment.json")
    latest_info = {
        "latest_deployment_dir": deploy_dir,
        "latest_model_path": deployed_model_path,
        "timestamp": timestamp,
        "model_name": model_name,
        "mlflow_model_name": model_version.name,
        "mlflow_model_version": model_version.version,
        "mlflow_model_stage": model_version.current_stage,
        "mlflow_run_id": run_id
    }
    
    with open(latest_config_path, "w") as f:
        json.dump(latest_info, f, indent=4)
    
    print(f"Model đã được triển khai tại: {deploy_dir}")
    print(f"Đây là triển khai mới nhất cho môi trường {deploy_env}")
    
    # Log triển khai vào MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("model_deployment")
    
    with mlflow.start_run(run_name=f"deploy-{model_name}-{deploy_env}-{timestamp}"):
        mlflow.log_param("deploy_env", deploy_env)
        mlflow.log_param("deployed_model_path", deployed_model_path)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_param("mlflow_model_name", model_version.name)
        mlflow.log_param("mlflow_model_version", model_version.version)
        mlflow.log_param("mlflow_model_stage", model_version.current_stage)
        mlflow.log_param("source_run_id", run_id)
        
        if config:
            for key, value in config.items():
                mlflow.log_param(f"config_{key}", value)
        
        # Log file cấu hình
        mlflow.log_artifact(config_path)
        
        deployment_run_id = mlflow.active_run().info.run_id
        print(f"Triển khai model đã được log vào MLflow với run_id: {deployment_run_id}")
    
    # Cập nhật thông tin triển khai trong run gốc
    try:
        client.set_tag(run_id, "deployed_to", deploy_env)
        client.set_tag(run_id, "deployment_timestamp", timestamp)
        client.set_tag(run_id, "deployment_path", deployed_model_path)
        client.set_tag(run_id, "deployment_run_id", deployment_run_id)
        print(f"Đã cập nhật thông tin triển khai trong run gốc: {run_id}")
    except Exception as e:
        print(f"Không thể cập nhật thông tin triển khai trong run gốc: {str(e)}")
    
    # Cập nhật thông tin môi trường triển khai trong model version
    try:
        client.set_model_version_tag(model_name, model_version.version, "deployed_to", deploy_env)
        client.set_model_version_tag(model_name, model_version.version, "deployment_timestamp", timestamp)
        client.set_model_version_tag(model_name, model_version.version, "deployment_path", deployed_model_path)
        client.set_model_version_tag(model_name, model_version.version, "deployment_run_id", deployment_run_id)
        print(f"Đã cập nhật thông tin triển khai trong model version {model_name}/{model_version.version}")
    except Exception as e:
        print(f"Không thể cập nhật thông tin triển khai trong model version: {str(e)}")

    return deployed_model_path

def list_deployments():
    """Liệt kê tất cả các triển khai model có sẵn"""
    # Tìm thư mục deployed_models
    deploy_dir = "deployed_models"
    
    if not os.path.exists(deploy_dir):
        print("Không tìm thấy thư mục triển khai.")
        return []
    
    # Lấy danh sách triển khai
    deployments = []
    for env_name in os.listdir(deploy_dir):
        env_dir = os.path.join(deploy_dir, env_name)
        if os.path.isdir(env_dir):
            # Kiểm tra xem đây có phải là thư mục môi trường hay thư mục triển khai
            if os.path.exists(os.path.join(env_dir, "deployment_config.json")):
                # Đây là thư mục triển khai cũ (trước khi cập nhật)
                with open(os.path.join(env_dir, "deployment_config.json"), "r") as f:
                    config = json.load(f)
                deployments.append(config)
            else:
                # Đây là thư mục môi trường, tìm các triển khai bên trong
                for deploy_name in os.listdir(env_dir):
                    deploy_subdir = os.path.join(env_dir, deploy_name)
                    if os.path.isdir(deploy_subdir):
                        config_path = os.path.join(deploy_subdir, "deployment_config.json")
                        if os.path.exists(config_path):
                            with open(config_path, "r") as f:
                                config = json.load(f)
                            deployments.append(config)
    
    # Sắp xếp triển khai theo thời gian (mới nhất trước)
    deployments.sort(key=lambda x: x["deployed_at"], reverse=True)
    
    # In danh sách triển khai
    if deployments:
        print("Danh sách triển khai model:")
        for i, deployment in enumerate(deployments):
            print(f"{i+1}. Môi trường: {deployment['environment']} - {deployment['deployed_at']}")
            print(f"   Model: {deployment.get('model_name', 'Không xác định')}")
            if 'mlflow_model_name' in deployment:
                print(f"   MLflow Model: {deployment['mlflow_model_name']} (Phiên bản: {deployment['mlflow_model_version']}, Giai đoạn: {deployment['mlflow_model_stage']})")
            print(f"   Đường dẫn triển khai: {deployment['deployed_model_path']}")
            if deployment.get("config"):
                print(f"   Cấu hình: {deployment['config']}")
            print()
    else:
        print("Không có triển khai model nào.")
    
    return deployments

def get_latest_deployment(deploy_env="production"):
    """Lấy thông tin về triển khai mới nhất trong môi trường cụ thể"""
    deploy_base_dir = os.path.join("deployed_models", deploy_env)
    latest_config_path = os.path.join(deploy_base_dir, "latest_deployment.json")
    
    if os.path.exists(latest_config_path):
        with open(latest_config_path, "r") as f:
            return json.load(f)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triển khai model từ MLflow Model Registry")
    subparsers = parser.add_subparsers(dest="command", help="Lệnh")
    
    # Parser cho lệnh deploy
    deploy_parser = subparsers.add_parser("deploy", help="Triển khai model từ MLflow Registry")
    deploy_parser.add_argument("--model-name", type=str, required=True, help="Tên model trong registry")
    deploy_parser.add_argument("--version", type=str, help="Phiên bản model cụ thể")
    deploy_parser.add_argument("--stage", type=str, choices=["Production", "Staging", "Archived", "None"], 
                             help="Giai đoạn của model")
    deploy_parser.add_argument("--env", type=str, default="production", choices=["production", "staging", "dev"], 
                             help="Môi trường triển khai")
    deploy_parser.add_argument("--config", type=str, help="Đường dẫn đến file cấu hình JSON")
    
    # Parser cho lệnh list
    list_parser = subparsers.add_parser("list", help="Liệt kê tất cả các triển khai")
    
    # Parser cho lệnh get-latest
    latest_parser = subparsers.add_parser("get-latest", help="Lấy thông tin triển khai mới nhất")
    latest_parser.add_argument("--env", type=str, default="production", choices=["production", "staging", "dev"], help="Môi trường triển khai")
    
    args = parser.parse_args()
    
    if args.command == "deploy":
        config = None
        if args.config:
            with open(args.config, "r") as f:
                config = json.load(f)
        deploy_model_from_registry(args.model_name, args.version, args.stage, args.env, config)
    elif args.command == "list":
        list_deployments()
    elif args.command == "get-latest":
        latest = get_latest_deployment(args.env)
        if latest:
            print(f"Triển khai mới nhất trong môi trường {args.env}:")
            for key, value in latest.items():
                print(f"  {key}: {value}")
        else:
            print(f"Không tìm thấy triển khai nào trong môi trường {args.env}")
    else:
        parser.print_help()