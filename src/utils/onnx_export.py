"""
ONNX export utilities for YOLO models.

Simple wrapper around Ultralytics built-in export functionality.
"""

from ultralytics import YOLO
from typing import Optional


def export_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    dynamic: bool = True,
    simplify: bool = True,
    opset: int = 12
) -> str:
    """
    Export YOLO model to ONNX format.
    
    This is a simple wrapper around Ultralytics' built-in export.
    Use this for programmatic export. For CLI, use:
        yolo export model=path/to/model.pt format=onnx
    
    Args:
        model_path: Path to PyTorch model (.pt)
        output_path: Optional output path (default: same dir as model)
        dynamic: Enable dynamic input shapes
        simplify: Simplify ONNX model
        opset: ONNX opset version
        
    Returns:
        Path to exported ONNX model
        
    Example:
        >>> export_to_onnx('models/train/weights/best.pt')
        'models/train/weights/best.onnx'
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Exporting to ONNX (dynamic={dynamic}, simplify={simplify})...")
    export_path = model.export(
        format="onnx",
        dynamic=dynamic,
        simplify=simplify,
        opset=opset
    )
    
    print(f"✅ Model exported successfully to: {export_path}")
    
    return str(export_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Output path (optional)")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic shapes")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        dynamic=not args.no_dynamic,
        simplify=not args.no_simplify,
        opset=args.opset
    )
