import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def register_model(run_id, model_name, stage, model_path=None, description=None, model_format=None):
    """
    Đăng ký model vào MLflow Model Registry
    
    Args:
        run_id (str): MLflow run ID
        model_name (str): Tên để đăng ký model
        model_path (str): Đường dẫn đến file model
        description (str): Mô tả cho phiên bản model
        stage (str): Giai đoạn cho model (Production, Staging, Archived, None)
        model_format (str): Định dạng model muốn đăng ký (onnx, pt, pth)
    """
    # Thiết lập MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # Lưu định dạng model vào biến môi trường để sử dụng sau
    if model_format:
        os.environ["MODEL_FORMAT"] = model_format
    
    
    # Nếu model_path được cung cấp, log nó như một artifact
    if model_path and os.path.exists(model_path):
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(model_path, "model")
    
    # Đăng ký model
    try:
        # Kiểm tra xem model đã tồn tại chưa
        try:
            model_details = client.get_registered_model(model_name)
            print(f"Model {model_name} đã tồn tại trong registry")
        except:
            # Tạo model nếu nó chưa tồn tại
            client.create_registered_model(model_name)
            print(f"Đã tạo model mới {model_name} trong registry")
        
        # Tạo phiên bản mới với định dạng cụ thể trong MLflow
        model_uri = f"runs:/{run_id}/model"
        
        # Kiểm tra định dạng mô hình
        model_format = os.environ.get("MODEL_FORMAT", "").lower()
        if model_format in ["onnx", "pt", "pth"]:
            # Kiểm tra xem có file mô hình với định dạng mong muốn không
            artifacts = client.list_artifacts(run_id, "model")
            format_exists = False
            
            for artifact in artifacts:
                if not artifact.is_dir and os.path.splitext(artifact.path)[1].lower() == f".{model_format}":
                    # Tìm thấy mô hình với định dạng mong muốn
                    model_uri = f"runs:/{run_id}/{artifact.path}"
                    format_exists = True
                    print(f"Đăng ký mô hình với định dạng {model_format}: {artifact.path}")
                    break
            
            if not format_exists:
                print(f"Cảnh báo: Không tìm thấy mô hình với định dạng {model_format}, sử dụng toàn bộ thư mục model")
        
        # Đăng ký mô hình với URI đã xác định
        model_version = mlflow.register_model(
            model_uri,
            model_name
        )
        
        print(f"Đã đăng ký model {model_name} phiên bản {model_version.version}")
        
        # Thêm mô tả cho phiên bản nếu được cung cấp
        if description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
            print(f"Đã thêm mô tả cho phiên bản {model_version.version}")
        
        # Chuyển đổi giai đoạn nếu được chỉ định
        if stage:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            print(f"Model {model_name} phiên bản {model_version.version} hiện đang ở giai đoạn {stage}")
        
        return model_version
    
    except Exception as e:
        print(f"Lỗi khi đăng ký model: {str(e)}")
        raise

def list_model_versions(model_name="brain_tumor_detector"):
    """
    Liệt kê tất cả các phiên bản của model
    
    Args:
        model_name (str): Tên model trong registry
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    try:
        # Lấy thông tin chi tiết về model
        model_details = client.get_registered_model(model_name)
        print(f"Model: {model_name}")
        
        # Liệt kê các phiên bản
        print("\nDanh sách phiên bản:")
        for version in model_details.latest_versions:
            print(f"Phiên bản: {version.version}")
            print(f"  Giai đoạn: {version.current_stage}")
            print(f"  Thời gian tạo: {version.creation_timestamp}")
            print(f"  Mô tả: {version.description or 'Không có mô tả'}")
            print(f"  Run ID: {version.run_id}")
            print()
            
    except Exception as e:
        print(f"Lỗi khi lấy thông tin model: {str(e)}")
        raise

def transition_model_stage(model_name, version, stage):
    """
    Chuyển đổi giai đoạn của phiên bản model
    
    Args:
        model_name (str): Tên model trong registry
        version (str): Phiên bản model
        stage (str): Giai đoạn mới (Production, Staging, Archived, None)
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} phiên bản {version} đã được chuyển sang giai đoạn {stage}")
    except Exception as e:
        print(f"Lỗi khi chuyển đổi giai đoạn: {str(e)}")
        raise

def get_model_details(model_name="brain_tumor_detector", version=None):
    """
    Lấy thông tin chi tiết về model và định dạng của nó
    
    Args:
        model_name (str): Tên model trong registry
        version (str): Phiên bản model cụ thể (nếu None, lấy tất cả các phiên bản)
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    try:
        # Lấy thông tin chi tiết về model
        model_details = client.get_registered_model(model_name)
        print(f"Model: {model_name}")
        
        versions_to_check = []
        if version:
            # Lấy phiên bản cụ thể
            model_version = client.get_model_version(model_name, version)
            versions_to_check = [model_version]
        else:
            # Lấy tất cả các phiên bản
            versions_to_check = model_details.latest_versions
        
        print("\nThông tin chi tiết:")
        for ver in versions_to_check:
            print(f"Phiên bản: {ver.version}")
            print(f"  Giai đoạn: {ver.current_stage}")
            print(f"  Thời gian tạo: {ver.creation_timestamp}")
            print(f"  Mô tả: {ver.description or 'Không có mô tả'}")
            print(f"  Run ID: {ver.run_id}")
            
            # Lấy thông tin về artifacts để xác định định dạng model
            run = mlflow.get_run(ver.run_id)
            artifacts_uri = run.info.artifact_uri
            
            # Kiểm tra thư mục model
            model_artifacts = client.list_artifacts(ver.run_id, "model")
            
            if model_artifacts:
                print("  Định dạng model:")
                for artifact in model_artifacts:
                    if artifact.is_dir:
                        print(f"    - Thư mục: {artifact.path}")
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
            
            # Lấy thông tin về môi trường triển khai từ tags
            tags = run.data.tags
            if 'deploy_env' in tags:
                print(f"  Môi trường triển khai: {tags['deploy_env']}")
            else:
                print(f"  Môi trường triển khai: Không xác định")
            
            print()
            
    except Exception as e:
        print(f"Lỗi khi lấy thông tin model: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quản lý model trong MLflow Model Registry")
    subparsers = parser.add_subparsers(dest="command", help="Lệnh")
    
    # Parser cho lệnh register
    register_parser = subparsers.add_parser("register", help="Đăng ký model mới")
    register_parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID")
    register_parser.add_argument("--model-name", type=str, help="Tên để đăng ký model")
    register_parser.add_argument("--model-path", type=str, help="Đường dẫn đến file model")
    register_parser.add_argument("--description", type=str, help="Mô tả cho phiên bản model")
    register_parser.add_argument("--stage", type=str, default="Staging",
                                choices=["Production", "Staging", "Archived", "None"], 
                                help="Giai đoạn cho model")
    register_parser.add_argument("--model-format", type=str, choices=["onnx", "pt", "pth"], 
                                help="Định dạng model muốn đăng ký (onnx, pt, pth)")
    
    # Parser cho lệnh list
    list_parser = subparsers.add_parser("list", help="Liệt kê các phiên bản model")
    list_parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Tên model trong registry")
    
    # Parser cho lệnh transition
    transition_parser = subparsers.add_parser("transition", help="Chuyển đổi giai đoạn của model")
    transition_parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Tên model trong registry")
    transition_parser.add_argument("--version", type=str, required=True, help="Phiên bản model")
    transition_parser.add_argument("--stage", type=str, required=True, 
                                  choices=["Production", "Staging", "Archived", "None"], 
                                  help="Giai đoạn mới")
    
    # Parser cho lệnh details
    details_parser = subparsers.add_parser("details", help="Lấy thông tin chi tiết về model")
    details_parser.add_argument("--model-name", type=str, default="brain_tumor_detector", help="Tên model trong registry")
    details_parser.add_argument("--version", type=str, help="Phiên bản model cụ thể (nếu không cung cấp, lấy tất cả)")
    
    args = parser.parse_args()
    
    if args.command == "register":
        register_model(args.run_id, args.model_name, args.stage, args.model_path, args.description, args.model_format)
    elif args.command == "list":
        list_model_versions(args.model_name)
    elif args.command == "transition":
        transition_model_stage(args.model_name, args.version, args.stage)
    elif args.command == "details":
        get_model_details(args.model_name, args.version)
    else:
        parser.print_help()