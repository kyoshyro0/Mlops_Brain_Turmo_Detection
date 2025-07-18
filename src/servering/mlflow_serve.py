import os
import argparse
import mlflow
import subprocess
import time
import webbrowser
import json
from mlflow.tracking import MlflowClient

def serve_model(model_name=None, model_version=None, model_uri=None, port=5001):
    """
    Phục vụ model thông qua MLflow Model Serving
    
    Args:
        model_name (str): Tên model trong registry
        model_version (str): Phiên bản model (None cho phiên bản production mới nhất)
        model_uri (str): URI trực tiếp đến model (thay thế cho model_name và model_version)
        port (int): Port để phục vụ model
    """
    # Thiết lập MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # Xác định model URI
    if model_uri is None:
        if model_name is None:
            # Sử dụng model mặc định từ thư mục triển khai
            deploy_dir = os.path.join("d:\\Workspaces\\Mlops_Brain_Turmo", "deployed_models", "production")
            if os.path.exists(deploy_dir):
                config_path = os.path.join(deploy_dir, "deployment_config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        if "deployed_model_path" in config:
                            model_uri = config["deployed_model_path"]
                            print(f"Sử dụng model đã triển khai: {model_uri}")
        else:
            # Lấy model từ registry
            if model_version is None:
                # Lấy phiên bản production mới nhất
                model_versions = client.get_latest_versions(model_name, stages=["Production"])
                if not model_versions:
                    raise ValueError(f"Không tìm thấy phiên bản production cho model {model_name}")
                model_version = model_versions[0].version
            
            model_uri = f"models:/{model_name}/{model_version}"
            print(f"Sử dụng model từ registry: {model_uri}")
    
    if model_uri is None:
        raise ValueError("Không thể xác định model URI. Vui lòng cung cấp model_name, model_uri hoặc đảm bảo có model đã triển khai.")
    
    # Khởi động MLflow Model Serving
    cmd = f"mlflow models serve -m {model_uri} -p {port} --no-conda"
    
    print(f"Khởi động MLflow Model Serving tại http://localhost:{port}")
    print(f"Lệnh: {cmd}")
    
    # Mở trình duyệt để kiểm tra API
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")
    
    # Chạy MLflow Model Serving
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phục vụ model thông qua MLflow Model Serving")
    parser.add_argument("--model-name", type=str, help="Tên model trong registry")
    parser.add_argument("--model-version", type=str, help="Phiên bản model (None cho phiên bản production mới nhất)")
    parser.add_argument("--model-uri", type=str, help="URI trực tiếp đến model (thay thế cho model_name và model_version)")
    parser.add_argument("--port", type=int, default=5001, help="Port để phục vụ model")
    
    args = parser.parse_args()
    serve_model(args.model_name, args.model_version, args.model_uri, args.port)