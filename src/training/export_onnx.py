from ultralytics import YOLO
import os
import argparse
import torch
import sys

def export_to_onnx(model_path, output_path=None):
    """
    Xuất mô hình YOLO sang định dạng ONNX sử dụng PyTorch
    
    Tham số:
        model_path (str): Đường dẫn đến mô hình PyTorch (.pt)
        output_path (str, tùy chọn): Đường dẫn để lưu mô hình ONNX
    """
    # Kiểm tra xem file mô hình có tồn tại không
    if not os.path.exists(model_path):
        # Thử tìm kiếm với đường dẫn tuyệt đối
        abs_model_path = os.path.join(os.getcwd(), model_path)
        if not os.path.exists(abs_model_path):
            # Thử tìm kiếm trong thư mục models
            alt_model_path = os.path.join('models', 'train', 'weights', os.path.basename(model_path))
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
            else:
                raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
        else:
            model_path = abs_model_path
    
    # Tải mô hình
    print(f"Đang tải mô hình từ {model_path}...")
    model = YOLO(model_path)
    
    # Đặt đường dẫn xuất mặc định nếu không được cung cấp
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(os.path.dirname(model_path), f"{base_name}.onnx")
    
    # Đảm bảo thư mục đích tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Thử sử dụng phương thức xuất tích hợp trước
        print("Đang thử xuất bằng phương thức của Ultralytics...")
        success = model.export(format="onnx", dynamic=True, simplify=True)
        
        if success:
            print(f"Mô hình đã được xuất thành công sang ONNX: {output_path}")
            return
        else:
            print("Xuất bằng phương thức tích hợp thất bại, đang thử phương thức thay thế...")
    except Exception as e:
        print(f"Xuất bằng phương thức tích hợp thất bại với lỗi: {str(e)}")
        print("Đang thử phương thức xuất thay thế...")
    
    # Phương thức thay thế sử dụng PyTorch trực tiếp
    try:
        # Lấy mô hình PyTorch
        pytorch_model = model.model
        
        # Tạo đầu vào giả
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Xuất sang ONNX
        print(f"Đang xuất mô hình sang {output_path}...")
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Mô hình đã được xuất thành công sang ONNX sử dụng PyTorch: {output_path}")
    except Exception as e:
        print(f"Phương thức xuất thay thế thất bại: {str(e)}")
        print("Vui lòng thử cài đặt phiên bản ONNX tương thích:")
        print("pip uninstall -y onnx onnxruntime")
        print("pip install protobuf==3.20.0")
        print("pip install onnx==1.13.0")
        print("pip install onnxruntime==1.14.0")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuất mô hình YOLO sang định dạng ONNX")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến mô hình PyTorch (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn để lưu mô hình ONNX")
    
    args = parser.parse_args()
    export_to_onnx(args.model, args.output)