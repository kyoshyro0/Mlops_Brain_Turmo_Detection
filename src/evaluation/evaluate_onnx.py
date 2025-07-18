import os
import argparse
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor detection ONNX model using Ultralytics")
    parser.add_argument("--model-path", type=str, default="models/train/weights/best.onnx", 
                        help="Path to the ONNX model")
    parser.add_argument("--data-yaml", type=str, default="configs/data_test.yaml", 
                        help="Path to the data YAML file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_onnx", 
                        help="Directory to save evaluation results")
    return parser.parse_args()

def evaluate_model(model_path, data_yaml, output_dir):
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Đánh giá model
    print(f"Evaluating model on {data_yaml}...")
    results = model.val(data=data_yaml)
    
    # Lấy metrics
    metrics = results.results_dict
    
    # Lưu kết quả vào file JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Tạo báo cáo đánh giá
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Brain Tumor Detection Model Evaluation\n")
        f.write("=====================================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_yaml}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}\n")
        f.write(f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}\n")
        f.write(f"Precision: {metrics.get('metrics/precision(B)', 0):.4f}\n")
        f.write(f"Recall: {metrics.get('metrics/recall(B)', 0):.4f}\n\n")
        
        f.write("Class-wise Metrics:\n")
        for i, cls in enumerate(model.names.values()):
            if f"metrics/precision({i})" in metrics:
                f.write(f"{cls}:\n")
                f.write(f"  Precision: {metrics.get(f'metrics/precision({i})', 0):.4f}\n")
                f.write(f"  Recall: {metrics.get(f'metrics/recall({i})', 0):.4f}\n")
                f.write(f"  mAP50: {metrics.get(f'metrics/mAP50({i})', 0):.4f}\n")
                f.write(f"  mAP50-95: {metrics.get(f'metrics/mAP50-95({i})', 0):.4f}\n\n")
    
    # Tạo biểu đồ
    create_visualizations(model.names, metrics, output_dir)
    
    print(f"Evaluation completed. Results saved to {output_dir}")
    return metrics

def create_visualizations(class_names, metrics, output_dir):
    # Tạo thư mục cho biểu đồ
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Lấy metrics cho từng lớp
    classes = list(class_names.values())
    precisions = []
    recalls = []
    map50s = []
    
    for i in range(len(classes)):
        precisions.append(metrics.get(f"metrics/precision({i})", 0))
        recalls.append(metrics.get(f"metrics/recall({i})", 0))
        map50s.append(metrics.get(f"metrics/mAP50({i})", 0))
    
    # Vẽ biểu đồ precision-recall
    plt.figure(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, precisions, width, label='Precision')
    plt.bar(x + width/2, recalls, width, label='Recall')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Precision and Recall by Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'precision_recall.png'))
    plt.close()
    
    # Vẽ biểu đồ mAP50
    plt.figure(figsize=(10, 6))
    plt.bar(x, map50s, color='blue')
    plt.axhline(y=metrics.get('metrics/mAP50(B)', 0), color='r', linestyle='-', 
                label=f'Mean mAP50: {metrics.get("metrics/mAP50(B)", 0):.4f}')
    
    plt.xlabel('Classes')
    plt.ylabel('mAP50')
    plt.title('mAP50 by Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'map50.png'))
    plt.close()

def main():
    args = parse_args()
    evaluate_model(args.model_path, args.data_yaml, args.output_dir)

if __name__ == "__main__":
    main()