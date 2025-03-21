from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, predict
import numpy as np
import uvicorn
import os
import time
from functools import lru_cache
import cv2
import hashlib

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

# Cache cho model để tránh tải lại
@lru_cache(maxsize=1)  # Sử dụng lru_cache để cache kết quả của hàm, chỉ lưu 1 kết quả
def get_model():
    
    best_onnx_path = os.path.join('models', 'train', 'weights', 'best.onnx')  # Đường dẫn đến model ONNX tốt nhất
    if os.path.exists(best_onnx_path):
        print("Loading ONNX model for optimized inference...")
        return load_model(best_onnx_path)
    
    # Tìm và tải model PyTorch tốt nhất nếu không có ONNX
    best_model_path = os.path.join('models', 'train', 'weights', 'best.pt')  # Đường dẫn đến model PyTorch tốt nhất
    
    # Kiểm tra xem model tốt nhất có tồn tại không
    if os.path.exists(best_model_path):
        return load_model(best_model_path)
    
    # Sử dụng model cuối cùng nếu model tốt nhất không tồn tại
    last_model_path = os.path.join('models', 'train', 'weights', 'last.pt')  # Đường dẫn đến model PyTorch cuối cùng
    if os.path.exists(last_model_path):
        return load_model(last_model_path)
    
    return load_model('yolo11s.pt')

# Khởi tạo model
model = get_model()

# Cache cho các dự đoán gần đây
prediction_cache = {}  # Khởi tạo dictionary để lưu cache
MAX_CACHE_SIZE = 50  # Kích thước tối đa của cache

@app.get("/")  # Định nghĩa endpoint GET "/"
async def root():
    return {"message": "Brain Tumor Detection API is running"}


@lru_cache(maxsize=1)  # Sử dụng lru_cache để cache kết quả của hàm, chỉ lưu 1 kết quả
def get_onnx_model(model_path):
    """Load ONNX model with caching"""
    return load_model(model_path)

@app.post("/predict")  # Định nghĩa endpoint POST "/predict"
async def predict_image(file: UploadFile = File(...), use_onnx: bool = False):
    """Endpoint for brain tumor detection in uploaded images.

    Args:
        file (UploadFile): Uploaded image file
        use_onnx (bool): Whether to use ONNX model for faster inference

    Returns:
        JSONResponse: Prediction results including detected tumors and confidence scores
    """
    try:
        start_time = time.time()
        
        # Kiểm tra loại file
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")
        
        # Đọc nội dung file
        contents = await file.read()
        
        # Tạo hash đơn giản cho cache
        file_hash = hashlib.md5(contents).hexdigest()  # Tạo hash MD5 từ nội dung file
        
        # Kiểm tra xem dự đoán có trong cache không
        if file_hash in prediction_cache: 
            print(f"Cache hit for {file.filename}")
            return JSONResponse(content={
                "predictions": prediction_cache[file_hash],  # Dự đoán từ cache
                "message": "Prediction retrieved from cache",  # Thông báo lấy từ cache
                "processing_time": time.time() - start_time  # Thời gian xử lý
            })
        
        # Xử lý ảnh bằng OpenCV
        nparr = np.frombuffer(contents, np.uint8)  # Chuyển đổi nội dung file thành mảng numpy
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Giải mã ảnh bằng OpenCV
        if image_np is None: 
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Chuyển đổi BGR sang RGB
        
        # Tải model phù hợp dựa trên tham số use_onnx
        if use_onnx:
            onnx_model_path = os.path.join('models', 'train', 'weights', 'best.onnx')
            if os.path.exists(onnx_model_path):
                model_to_use = get_onnx_model(onnx_model_path)
                model_type = "ONNX" 
            else:
                # Sử dụng model PyTorch
                model_to_use = model
                model_type = "PyTorch (ONNX not found)"
        else:
            model_to_use = model
            model_type = "PyTorch"
        
        # Thực hiện dự đoán với model đã chọn
        results = predict(model_to_use, image_np)
        
        # Xử lý kết quả
        predictions = []
        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                predictions.append({
                    "class": results[0].names[int(box.cls[0])],  # Tên class
                    "confidence": float(box.conf[0]),  # Độ tin cậy
                    "bbox": box.xyxy[0].tolist()  # Tọa độ bounding box
                })
        
        # Cập nhật cache (giữ kích thước tối đa)
        if len(prediction_cache) >= MAX_CACHE_SIZE:
            # Xóa entry cũ nhất
            oldest_key = next(iter(prediction_cache))
            prediction_cache.pop(oldest_key)
        prediction_cache[file_hash] = predictions  # Thêm dự đoán mới vào cache

        processing_time = time.time() - start_time  # Tính thời gian xử lý
        print(f"Processing time: {processing_time:.4f} seconds")  # In thời gian xử lý
        
        return JSONResponse(content={
            "predictions": predictions,
            "message": "Prediction completed successfully",
            "processing_time": processing_time,
            "model_type": model_type
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)