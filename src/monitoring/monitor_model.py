import os
import argparse
import json
import datetime
import numpy as np
import cv2
import mlflow
from model import load_model, predict
import glob

def monitor_model(model_path, test_data_dir, output_dir="monitoring_results"):
    """
    Monitor model performance on test data
    
    Args:
        model_path (str): Path to the model file
        test_data_dir (str): Directory containing test data
        output_dir (str): Directory to save monitoring results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("model_monitoring")
    
    # Load model
    model = load_model(model_path)
    
    # Get list of test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(glob.glob(os.path.join(test_data_dir, ext)))
    
    if not test_images:
        print(f"No test images found in {test_data_dir}")
        return
    
    # Initialize metrics
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": model_path,
        "num_images": len(test_images),
        "avg_confidence": 0,
        "detection_rate": 0,
        "processing_time": 0,
        "class_distribution": {}
    }
    
    total_confidence = 0
    total_detections = 0
    total_time = 0
    class_counts = {}
    
    with mlflow.start_run(run_name=f"monitoring-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        # Log model information
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("test_data_dir", test_data_dir)
        mlflow.log_param("num_test_images", len(test_images))
        
        # Process each image
        for img_path in test_images:
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read image {img_path}")
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Measure inference time
            start_time = datetime.datetime.now()
            results = predict(model, img)
            end_time = datetime.datetime.now()
            
            # Calculate processing time
            processing_time = (end_time - start_time).total_seconds()
            total_time += processing_time
            
            # Extract confidence scores and class distribution
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                confidences = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else []
                
                # Count class distribution
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                
                if len(confidences) > 0:
                    total_confidence += np.mean(confidences)
                    total_detections += 1
        
        # Calculate average metrics
        if total_detections > 0:
            metrics["avg_confidence"] = float(total_confidence / total_detections)
        metrics["detection_rate"] = float(total_detections / len(test_images))
        metrics["processing_time"] = float(total_time / len(test_images))
        metrics["class_distribution"] = class_counts
        
        # Log metrics to MLflow
        mlflow.log_metric("avg_confidence", metrics["avg_confidence"])
        mlflow.log_metric("detection_rate", metrics["detection_rate"])
        mlflow.log_metric("processing_time", metrics["processing_time"])
        
        for class_name, count in class_counts.items():
            mlflow.log_metric(f"class_{class_name}", count)
        
        # Save metrics to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Log metrics file as artifact
        mlflow.log_artifact(metrics_file)
        
        print(f"Monitoring results saved to {metrics_file}")
        print(f"Average confidence: {metrics['avg_confidence']:.4f}")
        print(f"Detection rate: {metrics['detection_rate']:.4f}")
        print(f"Average processing time: {metrics['processing_time']:.4f} seconds")
        print(f"Class distribution: {class_counts}")
        
        return metrics, run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor model performance on test data")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--test-data", type=str, required=True, help="Directory containing test data")
    parser.add_argument("--output-dir", type=str, default="monitoring_results", help="Directory to save monitoring results")
    
    args = parser.parse_args()
    monitor_model(args.model_path, args.test_data, args.output_dir)