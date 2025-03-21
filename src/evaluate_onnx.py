import os
import argparse
import numpy as np
import cv2
import json
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Định nghĩa các lớp
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor detection ONNX model")
    parser.add_argument("--model-path", type=str, default="models/best.onnx", 
                        help="Path to the best ONNX model")
    parser.add_argument("--test-data", type=str, default="data/test", 
                        help="Path to test data directory")
    parser.add_argument("--img-size", type=int, default=640, 
                        help="Image size for inference")
    parser.add_argument("--conf-thres", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--iou-thres", type=float, default=0.45, 
                        help="IoU threshold for NMS")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_onnx", 
                        help="Directory to save evaluation results")
    return parser.parse_args()

def preprocess_image(img_path, img_size):
    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    # Chuyển đổi BGR sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Lưu ảnh gốc để hiển thị
    original_img = img.copy()
    
    # Resize ảnh
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Chuẩn hóa ảnh
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Chuyển đổi HWC sang NCHW (batch, channels, height, width)
    img_transposed = img_norm.transpose(2, 0, 1)
    
    # Thêm chiều batch
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch, original_img

def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Tạo ONNX Runtime session
    session = ort.InferenceSession(model_path)
    
    # Lấy tên của input và output
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    return session, input_name, output_names

def postprocess(outputs, img_shape, conf_thres, iou_thres):
    # Xử lý đầu ra từ model ONNX
    predictions = outputs[0]
    
    # Chuyển đổi từ [1, 8400, 8] hoặc [1, 8, 8400] sang [8400, 8]
    if len(predictions.shape) == 3:
        if predictions.shape[1] == 8400 and predictions.shape[2] == len(CLASSES) + 5:
            predictions = predictions[0]  # [8400, 8]
        elif predictions.shape[1] == len(CLASSES) + 5 and predictions.shape[2] == 8400:
            predictions = predictions[0].transpose(1, 0)  # [8400, 8]
    
    # Lọc theo ngưỡng confidence
    mask = predictions[:, 4] >= conf_thres
    predictions = predictions[mask]
    
    if len(predictions) == 0:
        return []
    
    # Lấy tọa độ, confidence và class
    boxes = predictions[:, :4]  # x, y, w, h hoặc x1, y1, x2, y2
    scores = predictions[:, 4]
    class_ids = np.argmax(predictions[:, 5:], axis=1)
    
    # Chuyển đổi từ xywh sang xyxy nếu cần
    if boxes.shape[1] == 4:
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes = np.stack((x1, y1, x2, y2), axis=1)
    
    # Áp dụng NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    
    results = []
    for i in indices:
        if isinstance(i, np.ndarray):
            i = i.item()
        
        box = boxes[i].astype(int)
        score = float(scores[i])
        class_id = int(class_ids[i])
        
        results.append({
            "bbox": box.tolist(),
            "confidence": score,
            "class_id": class_id,
            "class": CLASSES[class_id]
        })
    
    return results

def calculate_iou(box1, box2):
    # Tính IoU giữa hai bounding box
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tính diện tích giao nhau
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Tính diện tích của hai box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Tính IoU
    iou = intersection / (area1 + area2 - intersection)
    
    return iou

def evaluate_model(model_path, test_data_dir, img_size, conf_thres, iou_thres, output_dir):
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading ONNX model from {model_path}...")
    session, input_name, output_names = load_onnx_model(model_path)
    
    # Tìm tất cả ảnh test và annotations
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(Path(test_data_dir).glob(f"images/{ext}")))
    
    print(f"Found {len(test_images)} test images")
    
    # Chuẩn bị biến để lưu kết quả
    all_predictions = []
    all_ground_truths = []
    all_image_paths = []
    
    # Duyệt qua từng ảnh test
    for img_path in tqdm(test_images, desc="Evaluating"):
        # Lấy đường dẫn đến file annotation tương ứng
        label_path = str(img_path).replace('images', 'labels').replace(img_path.suffix, '.txt')
        
        # Đọc ground truth từ file annotation
        ground_truths = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Chuyển đổi từ định dạng YOLO (x_center, y_center, width, height) sang (x1, y1, x2, y2)
                        x1 = (x_center - width / 2) * img_size
                        y1 = (y_center - height / 2) * img_size
                        x2 = (x_center + width / 2) * img_size
                        y2 = (y_center + height / 2) * img_size
                        
                        ground_truths.append({
                            "bbox": [x1, y1, x2, y2],
                            "class_id": class_id,
                            "class": CLASSES[class_id]
                        })
        
        # Tiền xử lý ảnh
        img_batch, original_img = preprocess_image(str(img_path), img_size)
        
        # Thực hiện dự đoán
        outputs = session.run(output_names, {input_name: img_batch})
        
        # Hậu xử lý kết quả
        predictions = postprocess(outputs, original_img.shape, conf_thres, iou_thres)
        
        # Lưu kết quả
        all_predictions.append(predictions)
        all_ground_truths.append(ground_truths)
        all_image_paths.append(str(img_path))
    
    # Tính toán các metrics
    print("Calculating metrics...")
    
    # Chuẩn bị dữ liệu cho confusion matrix
    y_true = []
    y_pred = []
    
    # Tính precision, recall cho từng lớp
    class_metrics = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in CLASSES}
    
    # Duyệt qua từng ảnh
    for i, (preds, gts) in enumerate(zip(all_predictions, all_ground_truths)):
        # Đánh dấu các ground truth đã được match
        matched_gts = [False] * len(gts)
        
        # Duyệt qua từng prediction
        for pred in preds:
            pred_class = pred["class"]
            pred_bbox = pred["bbox"]
            
            # Tìm ground truth phù hợp nhất
            best_iou = 0.5  # IoU threshold
            best_gt_idx = -1
            
            for j, gt in enumerate(gts):
                if matched_gts[j]:
                    continue  # Bỏ qua nếu ground truth đã được match
                
                if gt["class"] == pred_class:
                    iou = calculate_iou(pred_bbox, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            # Nếu tìm thấy ground truth phù hợp
            if best_gt_idx >= 0:
                matched_gts[best_gt_idx] = True
                class_metrics[pred_class]["TP"] += 1
                y_true.append(CLASSES.index(pred_class))
                y_pred.append(CLASSES.index(pred_class))
            else:
                class_metrics[pred_class]["FP"] += 1
                y_true.append(-1)  # Không có ground truth tương ứng
                y_pred.append(CLASSES.index(pred_class))
        
        # Đếm các ground truth không được match (False Negatives)
        for j, matched in enumerate(matched_gts):
            if not matched:
                gt_class = gts[j]["class"]
                class_metrics[gt_class]["FN"] += 1
                y_true.append(CLASSES.index(gt_class))
                y_pred.append(-1)  # Không có prediction tương ứng
    
    # Tính precision, recall, F1 cho từng lớp
    results = {}
    for cls in CLASSES:
        TP = class_metrics[cls]["TP"]
        FP = class_metrics[cls]["FP"]
        FN = class_metrics[cls]["FN"]
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": TP + FN
        }
    
    # Tính mAP (mean Average Precision)
    mAP = np.mean([results[cls]["precision"] for cls in CLASSES])
    
    # Lọc các giá trị -1 từ y_true và y_pred
    valid_indices = [(i, j) for i, (j, k) in enumerate(zip(y_true, y_pred)) if j != -1 and k != -1]
    if valid_indices:
        valid_y_true = [y_true[i] for i, _ in valid_indices]
        valid_y_pred = [y_pred[i] for _, i in valid_indices]
        
        # Tạo confusion matrix
        cm = confusion_matrix(valid_y_true, valid_y_pred, labels=range(len(CLASSES)))
        
        # Vẽ confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (ONNX Model)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    
    # Lưu kết quả vào file JSON
    evaluation_results = {
        "model_path": model_path,
        "test_data_dir": test_data_dir,
        "conf_threshold": conf_thres,
        "iou_threshold": iou_thres,
        "mAP": float(mAP),
        "class_metrics": results
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=4)

# Tạo báo cáo đánh giá
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Brain Tumor Detection ONNX Model Evaluation\n")
        f.write("========================================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test data: {test_data_dir}\n")
        f.write(f"Confidence threshold: {conf_thres}\n")
        f.write(f"IoU threshold: {iou_thres}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"mAP: {mAP:.4f}\n\n")
        
        f.write("Class-wise Metrics:\n")
        for cls in CLASSES:
            metrics = results[cls]
            f.write(f"{cls}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-score: {metrics['f1']:.4f}\n")
            f.write(f"  Support: {metrics['support']}\n\n")
    
    # Vẽ biểu đồ precision-recall
    plt.figure(figsize=(10, 6))
    for i, cls in enumerate(CLASSES):
        plt.bar(i, results[cls]["precision"], alpha=0.7, label="Precision" if i == 0 else "")
        plt.bar(i, results[cls]["recall"], alpha=0.5, label="Recall" if i == 0 else "")
    
    plt.xticks(range(len(CLASSES)), CLASSES)
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Precision and Recall by Class (ONNX Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall.png'))
    plt.close()
    
    # Vẽ biểu đồ F1-score
    plt.figure(figsize=(10, 6))
    f1_scores = [results[cls]["f1"] for cls in CLASSES]
    plt.bar(range(len(CLASSES)), f1_scores, color='blue')
    plt.axhline(y=np.mean(f1_scores), color='r', linestyle='-', label=f'Mean F1: {np.mean(f1_scores):.4f}')
    
    plt.xticks(range(len(CLASSES)), CLASSES)
    plt.xlabel('Classes')
    plt.ylabel('F1-score')
    plt.title('F1-score by Class (ONNX Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_scores.png'))
    plt.close()
    
    # Tạo một số hình ảnh minh họa
    visualization_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Chọn ngẫu nhiên một số ảnh để hiển thị
    num_samples = min(10, len(all_image_paths))
    sample_indices = np.random.choice(len(all_image_paths), num_samples, replace=False)
    
    for idx in sample_indices:
        img_path = all_image_paths[idx]
        predictions = all_predictions[idx]
        ground_truths = all_ground_truths[idx]
        
        # Đọc ảnh
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Vẽ ground truth (màu xanh lá)
        for gt in ground_truths:
            x1, y1, x2, y2 = [int(coord) for coord in gt["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT: {gt['class']}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Vẽ predictions (màu đỏ)
        for pred in predictions:
            if pred["confidence"] >= conf_thres:
                x1, y1, x2, y2 = [int(coord) for coord in pred["bbox"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"{pred['class']}: {pred['confidence']:.2f}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Lưu ảnh
        img_filename = os.path.basename(img_path)
        output_path = os.path.join(visualization_dir, f"vis_{img_filename}")
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    print(f"Evaluation completed. Results saved to {output_dir}")
    print(f"mAP: {mAP:.4f}")
    
    return evaluation_results

def main():
    args = parse_args()
    evaluate_model(
        args.model_path,
        args.test_data,
        args.img_size,
        args.conf_thres,
        args.iou_thres,
        args.output_dir
    )

if __name__ == "__main__":
    main()