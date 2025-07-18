import os
import argparse
import subprocess
import datetime
import json
import mlflow
from mlflow.tracking import MlflowClient

def run_pipeline(data_yaml, epochs=100, batch_size=16, img_size=640, 
                 register=True, deploy=True, monitor=True, 
                 deploy_env="production", test_data_dir=None):
    """
    Chạy pipeline MLOps đầy đủ: train, đăng ký, triển khai và giám sát model
    
    Args:
        data_yaml (str): Đường dẫn đến file cấu hình dữ liệu YAML
        epochs (int): Số lượng epoch huấn luyện
        batch_size (int): Kích thước batch cho huấn luyện
        img_size (int): Kích thước ảnh đầu vào
        register (bool): Có đăng ký model vào registry không
        deploy (bool): Có triển khai model không
        monitor (bool): Có giám sát model không
        deploy_env (str): Môi trường triển khai
        test_data_dir (str): Thư mục chứa dữ liệu test
    """
    # Thiết lập MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlops_pipeline")
    
    pipeline_start_time = datetime.datetime.now()
    
    with mlflow.start_run(run_name=f"pipeline-{pipeline_start_time.strftime('%Y%m%d-%H%M%S')}") as pipeline_run:
        pipeline_run_id = pipeline_run.info.run_id
        
        # Log tham số pipeline
        mlflow.log_param("data_yaml", data_yaml)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("register", register)
        mlflow.log_param("deploy", deploy)
        mlflow.log_param("monitor", monitor)
        mlflow.log_param("deploy_env", deploy_env)
        
        # 1. Huấn luyện model
        print("=== Bắt đầu huấn luyện model ===")
        train_cmd = [
            "python", "d:\\Workspaces\\Mlops_Brain_Turmo\\src\\train.py",
            "--data", data_yaml,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--img-size", str(img_size)
        ]
        
        print(f"Lệnh huấn luyện: {' '.join(train_cmd)}")
        train_process = subprocess.run(train_cmd, capture_output=True, text=True)
        
        if train_process.returncode != 0:
            print(f"Lỗi trong quá trình huấn luyện: {train_process.stderr}")
            mlflow.log_param("training_status", "failed")
            mlflow.log_text(train_process.stderr, "training_error.log")
            return False
        
        print(train_process.stdout)
        mlflow.log_param("training_status", "success")
        mlflow.log_text(train_process.stdout, "training_output.log")
        
        # Tìm run_id từ output huấn luyện
        train_output = train_process.stdout
        run_id = None
        for line in train_output.split('\n'):
            if "MLflow run ID:" in line:
                run_id = line.split("MLflow run ID:")[1].strip()
                break
        
        if not run_id:
            print("Không thể tìm thấy MLflow run ID từ output huấn luyện")
            mlflow.log_param("run_id_found", False)
        else:
            mlflow.log_param("run_id_found", True)
            mlflow.log_param("training_run_id", run_id)
        
        # Tìm đường dẫn đến model tốt nhất
        best_model_path = os.path.join("d:\\Workspaces\\Mlops_Brain_Turmo", "models", "train", "weights", "best.pt")
        if not os.path.exists(best_model_path):
            print(f"Không tìm thấy model tốt nhất tại {best_model_path}")
            mlflow.log_param("best_model_found", False)
            return False
        
        mlflow.log_param("best_model_found", True)
        mlflow.log_param("best_model_path", best_model_path)
        
        # 2. Đăng ký model vào registry
        if register and run_id:
            print("\n=== Bắt đầu đăng ký model ===")
            register_cmd = [
                "python", "d:\\Workspaces\\Mlops_Brain_Turmo\\src\\register_model.py",
                "--run-id", run_id,
                "--model-name", "brain_tumor_detector",
                "--model-path", best_model_path
            ]
            
            print(f"Lệnh đăng ký: {' '.join(register_cmd)}")
            register_process = subprocess.run(register_cmd, capture_output=True, text=True)
            
            if register_process.returncode != 0:
                print(f"Lỗi trong quá trình đăng ký model: {register_process.stderr}")
                mlflow.log_param("registration_status", "failed")
                mlflow.log_text(register_process.stderr, "registration_error.log")
            else:
                print(register_process.stdout)
                mlflow.log_param("registration_status", "success")
                mlflow.log_text(register_process.stdout, "registration_output.log")
        
        # 3. Triển khai model
        if deploy:
            print("\n=== Bắt đầu triển khai model ===")
            deploy_cmd = [
                "python", "d:\\Workspaces\\Mlops_Brain_Turmo\\src\\deploy_model.py",
                "deploy",
                "--model-path", best_model_path,
                "--env", deploy_env
            ]
            
            print(f"Lệnh triển khai: {' '.join(deploy_cmd)}")
            deploy_process = subprocess.run(deploy_cmd, capture_output=True, text=True)
            
            if deploy_process.returncode != 0:
                print(f"Lỗi trong quá trình triển khai model: {deploy_process.stderr}")
                mlflow.log_param("deployment_status", "failed")
                mlflow.log_text(deploy_process.stderr, "deployment_error.log")
            else:
                print(deploy_process.stdout)
                mlflow.log_param("deployment_status", "success")
                mlflow.log_text(deploy_process.stdout, "deployment_output.log")
        
        # 4. Giám sát model
        if monitor and test_data_dir:
            print("\n=== Bắt đầu giám sát model ===")
            monitor_cmd = [
                "python", "d:\\Workspaces\\Mlops_Brain_Turmo\\src\\monitor_model.py",
                "--model-path", best_model_path,
                "--test-data", test_data_dir,
                "--output-dir", "monitoring_results"
            ]
            
            print(f"Lệnh giám sát: {' '.join(monitor_cmd)}")
            monitor_process = subprocess.run(monitor_cmd, capture_output=True, text=True)
            
            if monitor_process.returncode != 0:
                print(f"Lỗi trong quá trình giám sát model: {monitor_process.stderr}")
                mlflow.log_param("monitoring_status", "failed")
                mlflow.log_text(monitor_process.stderr, "monitoring_error.log")
            else:
                print(monitor_process.stdout)
                mlflow.log_param("monitoring_status", "success")
                mlflow.log_text(monitor_process.stdout, "monitoring_output.log")
        
        # Hoàn thành pipeline
        pipeline_end_time = datetime.datetime.now()
        pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
        mlflow.log_metric("pipeline_duration_seconds", pipeline_duration)
        
        print(f"\n=== Pipeline hoàn thành trong {pipeline_duration:.2f} giây ===")
        print(f"Pipeline run ID: {pipeline_run_id}")
        
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline MLOps đầy đủ")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn đến file cấu hình dữ liệu YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch cho huấn luyện")
    parser.add_argument("--img-size", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--no-register", action="store_false", dest="register", help="Không đăng ký model vào registry")
    parser.add_argument("--no-deploy", action="store_false", dest="deploy", help="Không triển khai model")
    parser.add_argument("--no-monitor", action="store_false", dest="monitor", help="Không giám sát model")
    parser.add_argument("--deploy-env", type=str, default="production", choices=["production", "staging", "dev"], help="Môi trường triển khai")
    parser.add_argument("--test-data", type=str, help="Thư mục chứa dữ liệu test")
    
    args = parser.parse_args()
    
    # Kiểm tra xem test_data có được cung cấp khi monitor=True không
    if args.monitor and not args.test_data:
        parser.error("--test-data là bắt buộc khi giám sát model được bật")
    
    run_pipeline(
        args.data, 
        args.epochs, 
        args.batch_size, 
        args.img_size, 
        args.register, 
        args.deploy, 
        args.monitor, 
        args.deploy_env, 
        args.test_data
    )