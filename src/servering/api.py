from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
import os
import time
from functools import lru_cache
import cv2
import hashlib
import mlflow
import uuid
from datetime import datetime
import json
import sys

# Xác định thư mục gốc dự án dựa trên vị trí file hiện tại
current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.utils.model import load_model, predict

app = FastAPI( 
    title="Brain Tumor Detection API",
    description="API for detecting brain tumors using YOLOv11s model",
    version="1.0.0"
)

#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gốc
    allow_credentials=True,  # Cho phép gửi credentials
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các header
)

# Thiết lập MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("brain_tumor_detection_inference")

# Tìm model đã triển khai
def find_deployed_model(environment="production"):
    """Tìm model đã triển khai trong thư mục deployed_models với môi trường cụ thể
    
    Args:
        environment (str): Môi trường triển khai (mặc định: "production")
        
    Returns:
        str: Đường dẫn đến model đã triển khai hoặc None nếu không tìm thấy
    """
    deploy_dir = os.path.join(PROJECT_ROOT, "deployed_models", environment)
    
    if os.path.exists(deploy_dir):
        # Kiểm tra file cấu hình triển khai
        config_path = os.path.join(deploy_dir, "deployment_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                if "deployed_model_path" in config:
                    model_path = config["deployed_model_path"]
                    if os.path.exists(model_path):
                        return model_path, config.get("model_type", "Unknown")
    
    return None, None

# Cache cho model để tránh tải lại
@lru_cache(maxsize=1)  # Sử dụng lru_cache để cache kết quả của hàm, chỉ lưu 1 kết quả
def get_model(environment="production", force_pytorch=False):
    # Kiểm tra model đã triển khai
    deployed_model, model_type = find_deployed_model(environment)
    if deployed_model and not force_pytorch:
        print(f"Loading deployed model from {environment}: {deployed_model}")
        return load_model(deployed_model), f"Deployed ({environment})", model_type
    
    # Tìm và tải model ONNX tốt nhất
    best_onnx_path = os.path.join(PROJECT_ROOT, 'models', 'train', 'weights', 'best.onnx')
    if os.path.exists(best_onnx_path) and not force_pytorch:
        print("Loading ONNX model for optimized inference...")
        return load_model(best_onnx_path), "ONNX", "ONNX"
    
    # Tìm và tải model PyTorch tốt nhất nếu không có ONNX hoặc force_pytorch=True
    best_model_path = os.path.join(PROJECT_ROOT, 'models', 'train', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        print("Loading PyTorch best model...")
        return load_model(best_model_path), "PyTorch (best)", "PyTorch"
    
    # Sử dụng model cuối cùng nếu model tốt nhất không tồn tại
    last_model_path = os.path.join(PROJECT_ROOT, 'models', 'train', 'weights', 'last.pt')
    if os.path.exists(last_model_path):
        return load_model(last_model_path), "PyTorch (last)", "PyTorch"
    
    return load_model('yolo11s.pt'), "Default", "PyTorch"

# Khởi tạo model
model, model_source, model_type = get_model(force_pytorch=True)  # Mặc định sử dụng PyTorch
# Khởi tạo model ONNX cho trường hợp cần sử dụng
onnx_model, onnx_model_source, onnx_model_type = get_model(force_pytorch=False)

# Cache cho các dự đoán gần đây
prediction_cache = {}  # Khởi tạo dictionary để lưu cache
MAX_CACHE_SIZE = 50  # Kích thước tối đa của cache

@app.get("/")  # Định nghĩa endpoint GET "/"
async def root():
    return {"message": "Brain Tumor Detection API is running"}

@app.get("/model-info")
async def model_info(use_onnx: bool = False):
    """Endpoint để lấy thông tin về model đang được sử dụng"""
    deployed_model, deployed_type = find_deployed_model()
    
    if use_onnx:
        # Kiểm tra ONNX model
        onnx_model_path = os.path.join(PROJECT_ROOT, 'models', 'train', 'weights', 'best.onnx')
        if os.path.exists(onnx_model_path):
            info = {
                "model_source": "ONNX",
                "deployed_model": deployed_model if deployed_model and deployed_model.endswith('.onnx') else "None",
                "model_type": "ONNX"
            }
        else:
            info = {
                "model_source": model_source,
                "deployed_model": deployed_model if deployed_model else "None",
                "model_type": model_type,
                "note": "ONNX model not found, using PyTorch model instead"
            }
    else:
        info = {
            "model_source": model_source,
            "deployed_model": deployed_model if deployed_model else "None",
            "model_type": model_type
        }
    
    # Thêm thông tin từ file cấu hình triển khai nếu có
    if deployed_model:
        deploy_dir = os.path.dirname(deployed_model)
        config_path = os.path.join(deploy_dir, "deployment_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                deploy_config = json.load(f)
                info["deployment_info"] = deploy_config
    
    return JSONResponse(content=info)

@lru_cache(maxsize=1)  # Sử dụng lru_cache để cache kết quả của hàm, chỉ lưu 1 kết quả
def get_onnx_model(model_path):
    """Load ONNX model with caching"""
    return load_model(model_path), "ONNX"

@app.post("/predict")  # Định nghĩa endpoint POST "/predict"
async def predict_image(
    file: UploadFile = File(...), 
    use_onnx: bool = False,  # Mặc định là False
    confidence: float = 0.0,  # Thêm tham số confidence
    environment: str = "production"  # Thêm tham số môi trường
):
    """Endpoint for brain tumor detection in uploaded images.

    Args:
        file (UploadFile): Uploaded image file
        use_onnx (bool): Whether to use ONNX model for faster inference
        confidence (float): Confidence threshold for filtering predictions
        environment (str): Deployment environment to use (default: "production")

    Returns:
        JSONResponse: Prediction results including detected tumors and confidence scores
    """
    try:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Kiểm tra loại file
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
        
        # Đọc nội dung file
        contents = await file.read()
        
        # Tạo hash đơn giản cho cache
        file_hash = hashlib.md5(contents).hexdigest()  # Tạo hash MD5 từ nội dung file
        
        # Thêm confidence và environment vào cache key để phân biệt các ngưỡng và môi trường khác nhau
        cache_key = f"{file_hash}_{confidence}_{use_onnx}_{environment}"
        
        # Kiểm tra xem dự đoán có trong cache không
        if cache_key in prediction_cache: 
            print(f"Cache hit for {file.filename}")
            return JSONResponse(content={
                "predictions": prediction_cache[cache_key],  # Dự đoán từ cache
                "message": "Prediction retrieved from cache",  # Thông báo lấy từ cache
                "processing_time": time.time() - start_time,  # Thời gian xử lý
                "model_type": prediction_cache.get(f"{cache_key}_model_type", model_source),
                "model_framework": prediction_cache.get(f"{cache_key}_model_framework", model_type),
                "request_id": request_id
            })
        
        # Xử lý ảnh bằng OpenCV
        nparr = np.frombuffer(contents, np.uint8)  # Chuyển đổi nội dung file thành mảng numpy
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Giải mã ảnh bằng OpenCV
        if image_np is None: 
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Chuyển đổi BGR sang RGB
        
        # Tải model phù hợp dựa trên tham số use_onnx và environment
        model_to_use = None
        model_type_to_use = None
        model_framework = None
        
        if use_onnx:
            # Sử dụng ONNX model khi checkbox được chọn
            # Kiểm tra ONNX model trong môi trường triển khai
            deployed_model, deployed_type = find_deployed_model(environment)
            if deployed_model and deployed_model.endswith('.onnx'):
                model_to_use = load_model(deployed_model)
                model_type_to_use = f"Deployed ONNX ({environment})"
                model_framework = "ONNX"
            else:
                # Tìm ONNX model trong thư mục weights
                onnx_model_path = os.path.join(PROJECT_ROOT, 'models', 'train', 'weights', 'best.onnx')
                if os.path.exists(onnx_model_path):
                    model_to_use, _ = get_onnx_model(onnx_model_path)
                    model_type_to_use = "ONNX"
                    model_framework = "ONNX"
                else:
                    # Sử dụng model PyTorch nếu không tìm thấy ONNX
                    model_to_use = model  # Sử dụng model PyTorch mặc định
                    model_type_to_use = model_source
                    model_framework = model_type
                    print(f"ONNX model not found, falling back to {model_type}")
        else:
            # Sử dụng PyTorch model khi không chọn ONNX
            model_to_use = model  # Changed from pytorch_model to model
            model_type_to_use = model_source  # Changed from pytorch_model_source to model_source
            model_framework = model_type  # Changed from pytorch_model_type to model_type
            
            # Sử dụng model từ môi trường được chỉ định nếu khác với môi trường mặc định
            if environment != "production":
                env_model, env_model_source, env_model_type = get_model(environment, force_pytorch=True)
                model_to_use = env_model
                model_type_to_use = env_model_source
                model_framework = env_model_type
        
        # Thực hiện dự đoán với model đã chọn
        results = predict(model_to_use, image_np)
        
        # Xử lý kết quả
        predictions = []
        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                # Lọc theo ngưỡng confidence
                if conf >= confidence:
                    predictions.append({
                        "class": results[0].names[int(box.cls[0])],
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist()
                    })
        
        # Cập nhật cache (giữ kích thước tối đa)
        if len(prediction_cache) >= MAX_CACHE_SIZE:
            # Xóa entry cũ nhất
            oldest_key = next(iter(prediction_cache))
            prediction_cache.pop(oldest_key)
            # Xóa cả thông tin model type nếu có
            if f"{oldest_key}_model_type" in prediction_cache:
                prediction_cache.pop(f"{oldest_key}_model_type")
            if f"{oldest_key}_model_framework" in prediction_cache:
                prediction_cache.pop(f"{oldest_key}_model_framework")
                
        prediction_cache[cache_key] = predictions  # Thêm dự đoán mới vào cache
        prediction_cache[f"{cache_key}_model_type"] = model_type_to_use  # Lưu loại model
        prediction_cache[f"{cache_key}_model_framework"] = model_framework  # Lưu framework

        processing_time = time.time() - start_time  # Tính thời gian xử lý
        
        # Log dự đoán với MLflow
        with mlflow.start_run(run_name=f"inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log thông tin request
            mlflow.log_param("request_id", request_id)
            mlflow.log_param("model_type", model_type_to_use)
            mlflow.log_param("model_framework", model_framework)
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("environment", environment)
            
            # Log metrics
            mlflow.log_metric("processing_time", processing_time)
            mlflow.log_metric("num_detections", len(predictions))
            
            if len(predictions) > 0:
                # Log confidence scores
                confidences = [pred["confidence"] for pred in predictions]
                mlflow.log_metric("avg_confidence", sum(confidences) / len(confidences))
                mlflow.log_metric("max_confidence", max(confidences))
        
        print(f"Processing time: {processing_time:.4f} seconds")  # In thời gian xử lý
        
        return JSONResponse(content={
            "predictions": predictions,
            "message": "Prediction completed successfully",
            "processing_time": processing_time,
            "model_type": model_type_to_use,
            "model_framework": model_framework,
            "request_id": request_id
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Error during prediction: {str(e)}",
                "predictions": []
            }
        )

@app.get("/mlflow-dashboard")
async def mlflow_dashboard():
    """Endpoint để lấy URL của MLflow dashboard"""
    mlflow_url = "http://localhost:5000"
    return JSONResponse(content={
        "mlflow_url": mlflow_url,
        "message": "Đảm bảo MLflow UI đang chạy bằng cách thực thi 'python src/mlflow_dashboard.py'"
    })

if __name__ == "__main__":
    # When running directly, use the local reference
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)