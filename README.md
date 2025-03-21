# Brain Tumor Detection

Ứng dụng phát hiện khối u não sử dụng mô hình YOLOv11s.

## Tính năng

- Phát hiện 4 loại khối u não: glioma, meningioma, pituitary và notumor
- Hỗ trợ mô hình ONNX để tăng tốc độ suy luận
- Giao diện người dùng thân thiện với Streamlit
- API backend sử dụng FastAPI
- Hỗ trợ Docker để dễ dàng triển khai

## Cài đặt

### Yêu cầu

- Python 3.10+
- Docker (tùy chọn)

### Cài đặt thủ công

1. Clone repository:
```bash
git clone https://github.com/username/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.backend.txt
pip install -r requirements.frontend.txt
```
3. Chạy ứng dụng:
```bash
# Chạy backend API
python src/api.py

# Trong terminal khác, chạy frontend
streamlit run src/app.py
```

### Sử dụng Docker
```bash
docker-compose -f docker/docker-compose.yml build --no-cache
docker-compose -f docker/docker-compose.yml up -d
```

### Sử dụng
1. Mở trình duyệt và truy cập http://localhost:8501
2. Tải lên ảnh MRI não
3. Điều chỉnh ngưỡng tin cậy nếu cần
4. Xem kết quả phát hiện
