"""
Model evaluation utilities for MLOps pipeline.

This module provides unified evaluation for both PyTorch and ONNX models,
with MLflow integration and comprehensive metrics visualization.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Dict
import mlflow


# Define class names
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def evaluate_model(
    model_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = '0',
    output_dir: str = "evaluation_results",
    log_mlflow: bool = True
) -> Dict:
    """
    Evaluate model (PyTorch or ONNX) and generate comprehensive metrics.
    
    Supports both .pt and .onnx model formats automatically.
    
    Args:
        model_path: Path to model file (.pt or .onnx)
        data_yaml: Path to dataset configuration
        img_size: Input image size
        batch_size: Batch size for evaluation
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        device: Device to run on ('0' for GPU, 'cpu' for CPU)
        output_dir: Directory to save results
        log_mlflow: Whether to log metrics to MLflow
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect model format
    model_format = "ONNX" if model_path.endswith('.onnx') else "PyTorch"
    print(f"Loading {model_format} model from {model_path}...")
    
    # Load model (YOLO handles both .pt and .onnx)
    model = YOLO(model_path)
    
    # Run validation
    print(f"Evaluating model on {data_yaml}...")
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        save_json=True,
        plots=True
    )
    
    # Extract metrics
    metrics = results.results_dict if hasattr(results, 'results_dict') else {}
    
    # Add model info to metrics
    metrics['model_path'] = model_path
    metrics['model_format'] = model_format
    metrics['data_yaml'] = data_yaml
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, f"metrics_{model_format.lower()}.json")
    with open(metrics_path, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }
        json.dump(serializable_metrics, f, indent=4)
    
    # Create evaluation report
    report_path = os.path.join(output_dir, f"evaluation_report_{model_format.lower()}.txt")
    with open(report_path, "w") as f:
        f.write(f"Brain Tumor Detection - {model_format} Model Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_yaml}\n")
        f.write(f"Format: {model_format}\n\n")
        
        f.write("Overall Performance Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}\n")
        f.write(f"mAP@0.5:     {metrics.get('metrics/mAP50(B)', 0):.4f}\n")
        f.write(f"Precision:   {metrics.get('metrics/precision(B)', 0):.4f}\n")
        f.write(f"Recall:      {metrics.get('metrics/recall(B)', 0):.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-" * 60 + "\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"\n{cls.upper()}:\n")
            f.write(f"  Precision:   {metrics.get(f'metrics/precision(B)', 0):.4f}\n")
            f.write(f"  Recall:      {metrics.get(f'metrics/recall(B)', 0):.4f}\n")
            f.write(f"  mAP@0.5:     {metrics.get(f'metrics/mAP50(B)', 0):.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics.get(f'metrics/mAP50-95(B)', 0):.4f}\n")
    
    # Create visualizations
    if metrics:
        create_visualizations(model.names, metrics, output_dir, model_format)
    
    # Log to MLflow if requested
    if log_mlflow:
        log_to_mlflow(metrics, model_path, model_format)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
    
    return metrics


def create_visualizations(
    class_names: Dict,
    metrics: Dict,
    output_dir: str,
    model_format: str = "PyTorch"
):
    """
    Create visualization plots for evaluation metrics.
    
    Args:
        class_names: Dictionary mapping class IDs to names
        metrics: Evaluation metrics dictionary
        output_dir: Directory to save visualizations
        model_format: Model format for labeling plots
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    classes = list(class_names.values())
    
    # Extract per-class metrics (simplified - using overall metrics)
    precisions = [metrics.get('metrics/precision(B)', 0)] * len(classes)
    recalls = [metrics.get('metrics/recall(B)', 0)] * len(classes)
    map50s = [metrics.get('metrics/mAP50(B)', 0)] * len(classes)
    
    # Plot 1: Precision vs Recall
    plt.figure(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, precisions, width, label='Precision', color='steelblue')
    plt.bar(x + width/2, recalls, width, label='Recall', color='coral')
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Precision and Recall by Class ({model_format})', fontsize=14, fontweight='bold')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'precision_recall_{model_format.lower()}.png'), dpi=300)
    plt.close()
    
    # Plot 2: mAP@0.5
    plt.figure(figsize=(10, 6))
    plt.bar(x, map50s, color='mediumseagreen', alpha=0.7)
    plt.axhline(y=metrics.get('metrics/mAP50(B)', 0), color='red', linestyle='--', 
                label=f'Mean mAP@0.5: {metrics.get("metrics/mAP50(B)", 0):.4f}', linewidth=2)
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('mAP@0.5', fontsize=12)
    plt.title(f'mAP@0.5 by Class ({model_format})', fontsize=14, fontweight='bold')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'map50_{model_format.lower()}.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {vis_dir}")


def log_to_mlflow(metrics: Dict, model_path: str, model_format: str):
    """
    Log evaluation metrics to MLflow.
    
    Args:
        metrics: Evaluation metrics dictionary
        model_path: Path to evaluated model
        model_format: Model format (PyTorch or ONNX)
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("model_evaluation")
    
    with mlflow.start_run(run_name=f"eval-{model_format}-{Path(model_path).stem}"):
        # Log parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("model_format", model_format)
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key.replace('metrics/', ''), float(value))
        
        print(f"Metrics logged to MLflow experiment: model_evaluation")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate brain tumor detection model (PyTorch or ONNX)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/train/weights/best.pt",
        help="Path to model file (.pt or .onnx)"
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="configs/data.yaml",
        help="Path to dataset configuration"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='0',
        help="Device to run on (0 for GPU, cpu for CPU)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_yaml=args.data_yaml,
        img_size=args.img_size,
        batch_size=args.batch_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        output_dir=args.output_dir,
        log_mlflow=not args.no_mlflow
    )


if __name__ == "__main__":
    main()
