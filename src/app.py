import streamlit as st
import os
import numpy as np
import requests
from PIL import Image, ImageDraw
import io
import hashlib

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🔍",
    layout="wide"
)

# Endpoint API - lấy từ biến môi trường hoặc sử dụng giá trị mặc định
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Tiêu đề và mô tả
st.title("Brain Tumor Detection")
st.markdown("Upload an image to detect brain tumors using YOLOv11s model")

# Thanh bên
st.sidebar.title("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
use_onnx = st.sidebar.checkbox("Use ONNX model (faster inference)", value=True)

# Khởi tạo session state để lưu cache
if 'prediction_cache' not in st.session_state:
    st.session_state.prediction_cache = {}

# Kiểm tra xem API có đang chạy không
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        st.success("✅ API is running and ready for predictions")
    else:
        st.error("❌ API is not responding correctly")
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to API. Please make sure the API server is running.")

# Công cụ upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tạo khóa cache dựa trên nội dung file và cài đặt
    file_bytes = uploaded_file.getvalue()
    settings_str = f"{confidence}_{use_onnx}"
    cache_key = hashlib.md5(file_bytes + settings_str.encode()).hexdigest()
    
    # Chuyển đổi file đã upload thành ảnh
    image = Image.open(uploaded_file)
    
    # Resize ảnh trước khi gửi đến API để giảm thời gian truyền
    max_size = 640  # Kích thước tối đa phù hợp với YOLO
    orig_width, orig_height = image.size
    
    # Chỉ resize nếu ảnh lớn hơn kích thước tối đa
    if max(orig_width, orig_height) > max_size:
        if orig_width > orig_height:
            new_width = max_size
            new_height = int(orig_height * (max_size / orig_width))
        else:
            new_height = max_size
            new_width = int(orig_width * (max_size / orig_height))
        
        # Lưu tỷ lệ để khôi phục bounding box sau này
        scale_x = orig_width / new_width
        scale_y = orig_height / new_height
        
        # Resize ảnh
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        image_resized = image
        scale_x = scale_y = 1.0
    
    # Tạo cột cho ảnh gốc và ảnh đã xử lý
    col1, col2 = st.columns(2)
    
    # Hiển thị ảnh gốc trong cột 1
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Kiểm tra cache nhưng không hiển thị thông báo ở đây
    if cache_key in st.session_state.prediction_cache:
        predictions = st.session_state.prediction_cache[cache_key]
        error_message = None
        is_from_cache = True
    else:
        is_from_cache = False
        # Hiển thị spinner trong khi đang xử lý
        with st.spinner('Detecting tumors...'):
            # Chuẩn bị ảnh cho API request - tối ưu định dạng
            img_byte_arr = io.BytesIO()
            
            # Chuyển đổi từ RGBA sang RGB nếu cần
            if image_resized.mode == 'RGBA':
                image_resized_rgb = image_resized.convert('RGB')
            else:
                image_resized_rgb = image_resized
                
            image_resized_rgb.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr = img_byte_arr.getvalue()

            try:
                # Tạo request dự đoán đến API với tham số ONNX
                files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                params = {"use_onnx": "true" if use_onnx else "false"}
                
                # Đo thời gian phản hồi
                import time
                start_time = time.time()
                
                response = requests.post(f"{API_URL}/predict", files=files, params=params)
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Kiểm tra xem có thông báo lỗi trong phản hồi không
                    error_message = response_data.get("error", None)
                    if "detail" in response_data:
                        error_message = response_data["detail"]
                    
                    predictions = response_data.get("predictions", [])
                    
                    # Lưu kết quả vào cache chỉ khi không có lỗi
                    if not error_message:
                        st.session_state.prediction_cache[cache_key] = predictions
                else:
                    st.error(f"Error from API: {response.text}")
                    predictions = []
                    error_message = response.text
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with API: {str(e)}")
                predictions = []
                error_message = str(e)
    
    # Tạo bản sao của ảnh để vẽ lên
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # Xử lý kết quả và vẽ lên ảnh
    if predictions:
        for pred in predictions:
            if pred["confidence"] >= confidence:
                # Lấy tọa độ bounding box và scale lại về kích thước ảnh gốc
                bbox = pred["bbox"]
                x1, y1, x2, y2 = bbox
                
                # Scale tọa độ về kích thước ảnh gốc nếu đã resize
                if scale_x != 1.0 or scale_y != 1.0:
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                
                x1, y1, x2, y2 = [int(coord) for coord in [x1, y1, x2, y2]]
                
                # Vẽ hình chữ nhật lên ảnh
                # Màu dựa trên class (có thể tùy chỉnh)
                colors = {
                    "glioma": (255, 0, 0),      # Đỏ
                    "meningioma": (0, 255, 0),  # Xanh lá
                    "pituitary": (0, 0, 255),   # Xanh dương
                    "notumor": (255, 255, 0)    # Vàng
                }
                color = colors.get(pred["class"], (255, 0, 0))
                
                # Vẽ bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Thêm nhãn
                label = f"{pred['class']}: {pred['confidence']:.2f}"
                draw.text((x1, y1-15), label, fill=color)
    
    # Hiển thị ảnh kết quả trong cột 2
    with col2:
        st.subheader("Detection Results")
        if predictions:
            st.image(result_image, caption="Detection Results", use_container_width=True)
        else:
            st.info("No tumors detected in the image")
            st.image(image, caption="No tumors detected", use_container_width=True)
    
    # Hiển thị thông tin kết quả bên dưới cả hai ảnh (ngoài columns)
    st.subheader("Detection Information")
    
    # Hiển thị thông báo cache ở đây thay vì ở trên
    if is_from_cache:
        st.success("✅ Retrieved result from cache")
    else:
        # Hiển thị thời gian xử lý và thông tin model
        if 'processing_time' in locals():
            st.info(f"Processing time: {processing_time:.3f} seconds")
        
        if 'response_data' in locals() and "model_type" in response_data:
            st.success(f"Used {response_data['model_type']} model for inference")
    
    # Hiển thị kết quả chi tiết
    if predictions:
        st.subheader("Detection Details")
        for pred in predictions:
            if pred["confidence"] >= confidence:
                st.write(f"Class: {pred['class']} | Confidence: {pred['confidence']:.2f}")
    
    # Hiển thị thông báo lỗi nếu có
    if 'error_message' in locals() and error_message:
        st.error(f"Error from API: {error_message}")
