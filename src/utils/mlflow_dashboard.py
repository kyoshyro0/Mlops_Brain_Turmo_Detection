import os
import argparse
import subprocess
import webbrowser
import time

def start_mlflow_ui(host="127.0.0.1", port=5000, open_browser=True):
    """
    Khởi động MLflow UI
    
    Args:
        host (str): Host để chạy MLflow UI
        port (int): Port để chạy MLflow UI
        open_browser (bool): Có mở trình duyệt tự động không
    """
    # Đảm bảo thư mục mlruns tồn tại
    os.makedirs("mlruns", exist_ok=True)
    
    # Khởi động MLflow UI
    cmd = f"mlflow ui --host {host} --port {port} --backend-store-uri sqlite:///mlflow.db"
    
    print(f"Khởi động MLflow UI tại http://{host}:{port}")
    print(f"Lệnh: {cmd}")
    
    # Mở trình duyệt nếu được yêu cầu
    if open_browser:
        # Đợi một chút để MLflow UI khởi động
        time.sleep(2)
        webbrowser.open(f"http://{host}:{port}")
    
    # Chạy MLflow UI
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khởi động MLflow UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host để chạy MLflow UI")
    parser.add_argument("--port", type=int, default=5000, help="Port để chạy MLflow UI")
    parser.add_argument("--no-browser", action="store_false", dest="open_browser", help="Không mở trình duyệt tự động")
    
    args = parser.parse_args()
    start_mlflow_ui(args.host, args.port, args.open_browser)