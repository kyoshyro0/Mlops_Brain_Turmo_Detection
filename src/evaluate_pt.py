import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO

# Định nghĩa các lớp
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor detection YOLOv11 model")
    parser.add_argument("--model-path", type=str, default="models/best.pt", 
                        help="Path to the best YOLOv11 model")
    parser.add_argument("--data-yaml", type=str, default="data/data.yaml", 
                        help="Path to data YAML file")
    parser.add_argument("--img-size", type=int, default=640, 
                        help="Image size for inference")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Batch size for validation")
    parser.add_argument("--conf-thres", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--iou-thres", type=float, default=0.45, 
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="", 
                        help="Device to run on (empty for auto, 'cpu', '0', '0,1,2,3')")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_pt", 
                        help="Directory to save evaluation results")
    return parser.parse_args()

def evaluate_model(model_path, data_yaml, img_size, batch_size, conf_thres, iou_thres, device, output_dir):
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Sử dụng đường dẫn trực tiếp đến model mà không cần kiểm tra
    model_path_abs = os.path.abspath(model_path)
    print(f"Using model at: {model_path_abs}")
    
    # Sử dụng đường dẫn trực tiếp đến data.yaml mà không cần kiểm tra
    data_yaml_abs = os.path.abspath(data_yaml)
    print(f"Using data YAML at: {data_yaml_abs}")
    
    # Load model sử dụng YOLOv11 API
    print(f"Loading YOLOv11 model from {model_path_abs}...")
    try:
        model = YOLO(model_path_abs)
    except Exception as e:
        print(f"Error loading YOLOv11 model: {str(e)}")
        raise
    
    # Thực hiện đánh giá model
    print(f"Evaluating model on {data_yaml_abs}...")
    try:
        # Sử dụng phương thức val() của YOLOv11
        results = model.val(
            data=data_yaml_abs,
            imgsz=img_size,
            batch=batch_size,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            verbose=True,
            save_json=True,
            save_hybrid=True,
            plots=True
        )
        
        # Lưu kết quả đánh giá
        metrics = results.box
        
        # Tạo dictionary để lưu kết quả
        evaluation_results = {
            "model_path": model_path,
            "data_yaml": data_yaml,
            "conf_threshold": conf_thres,
            "iou_threshold": iou_thres,
            "metrics": {
                "mAP50": float(metrics.map50) if not isinstance(metrics.map50, (list, np.ndarray)) else float(metrics.map50[0]),
                "mAP50-95": float(metrics.map) if not isinstance(metrics.map, (list, np.ndarray)) else float(metrics.map[0]),
                "precision": float(metrics.p) if not isinstance(metrics.p, (list, np.ndarray)) else float(metrics.p[0]),
                "recall": float(metrics.r) if not isinstance(metrics.r, (list, np.ndarray)) else float(metrics.r[0]),
                "f1": float(metrics.f1) if not isinstance(metrics.f1, (list, np.ndarray)) else float(metrics.f1[0]),
                "class_metrics": {}
            }
        }
        
        # Lưu metrics cho từng class
        if hasattr(metrics, 'ap_class_index') and hasattr(metrics, 'ap50'):
            for i, class_idx in enumerate(metrics.ap_class_index):
                if class_idx < len(CLASSES):
                    class_name = CLASSES[class_idx]
                    evaluation_results["metrics"]["class_metrics"][class_name] = {
                        "precision": float(metrics.p_per_class[i] if hasattr(metrics, 'p_per_class') and i < len(metrics.p_per_class) else 0),
                        "recall": float(metrics.r_per_class[i] if hasattr(metrics, 'r_per_class') and i < len(metrics.r_per_class) else 0),
                        "ap50": float(metrics.ap50[i] if i < len(metrics.ap50) else 0),
                        "ap": float(metrics.ap[i] if i < len(metrics.ap) else 0)
                    }
        
        # Lưu kết quả vào file JSON
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        # Tạo báo cáo đánh giá
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("Brain Tumor Detection YOLOv11 Model Evaluation\n")
            f.write("==========================================\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Data YAML: {data_yaml}\n")
            f.write(f"Confidence threshold: {conf_thres}\n")
            f.write(f"IoU threshold: {iou_thres}\n\n")
            
            f.write("Performance Metrics:\n")
            # Xử lý các giá trị có thể là mảng
            map50_val = float(metrics.map50) if not isinstance(metrics.map50, (list, np.ndarray)) else float(metrics.map50[0])
            map_val = float(metrics.map) if not isinstance(metrics.map, (list, np.ndarray)) else float(metrics.map[0])
            p_val = float(metrics.p) if not isinstance(metrics.p, (list, np.ndarray)) else float(metrics.p[0])
            r_val = float(metrics.r) if not isinstance(metrics.r, (list, np.ndarray)) else float(metrics.r[0])
            f1_val = float(metrics.f1) if not isinstance(metrics.f1, (list, np.ndarray)) else float(metrics.f1[0])
            
            f.write(f"mAP@0.5: {map50_val:.4f}\n")
            f.write(f"mAP@0.5:0.95: {map_val:.4f}\n")
            f.write(f"Precision: {p_val:.4f}\n")
            f.write(f"Recall: {r_val:.4f}\n")
            f.write(f"F1-score: {f1_val:.4f}\n\n")
            
            f.write("Class-wise Metrics:\n")
            if hasattr(metrics, 'ap_class_index') and hasattr(metrics, 'ap50'):
                for i, class_idx in enumerate(metrics.ap_class_index):
                    if class_idx < len(CLASSES):
                        class_name = CLASSES[class_idx]
                        f.write(f"{class_name}:\n")
                        if hasattr(metrics, 'p_per_class') and i < len(metrics.p_per_class):
                            f.write(f"  Precision: {metrics.p_per_class[i]:.4f}\n")
                        if hasattr(metrics, 'r_per_class') and i < len(metrics.r_per_class):
                            f.write(f"  Recall: {metrics.r_per_class[i]:.4f}\n")
                        if i < len(metrics.ap50):
                            f.write(f"  AP@0.5: {metrics.ap50[i]:.4f}\n")
                        if i < len(metrics.ap):
                            f.write(f"  AP@0.5:0.95: {metrics.ap[i]:.4f}\n")
                        f.write("\n")
        
        # Sao chép các biểu đồ từ thư mục runs/val vào output_dir
        val_dir = Path('runs/val')
        if val_dir.exists():
            latest_val = max([d for d in val_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
            for plot_file in latest_val.glob('*.png'):
                import shutil
                shutil.copy(plot_file, os.path.join(output_dir, plot_file.name))
        
        # Tạo biểu đồ bổ sung nếu có dữ liệu class-wise
        if hasattr(metrics, 'ap_class_index') and hasattr(metrics, 'ap50'):
            class_names = []
            ap50_values = []
            
            for i, class_idx in enumerate(metrics.ap_class_index):
                if class_idx < len(CLASSES) and i < len(metrics.ap50):
                    class_names.append(CLASSES[class_idx])
                    ap50_values.append(metrics.ap50[i])
            
            if class_names:
                plt.figure(figsize=(10, 6))
                plt.bar(class_names, ap50_values, color='blue')
                plt.axhline(y=metrics.map50, color='r', linestyle='-', label=f'mAP@0.5: {metrics.map50:.4f}')
                
                plt.xlabel('Classes')
                plt.ylabel('AP@0.5')
                plt.title('AP@0.5 by Class (YOLOv11 Model)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'ap50_by_class.png'))
                plt.close()
        
        print(f"Evaluation completed. Results saved to {output_dir}")
        print(f"mAP@0.5: {metrics.map50:.4f}, mAP@0.5:0.95: {metrics.map:.4f}")
        
        return evaluation_results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

def main():
    args = parse_args()
    evaluate_model(
        args.model_path,
        args.data_yaml,
        args.img_size,
        args.batch_size,
        args.conf_thres,
        args.iou_thres,
        args.device,
        args.output_dir
    )

if __name__ == "__main__":
    main()