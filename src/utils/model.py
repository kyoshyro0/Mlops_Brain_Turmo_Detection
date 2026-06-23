"""
Model utilities for YOLOv11 inference and training.

This module provides model loading, prediction, and training functionality
with support for both PyTorch and ONNX Runtime backends.
"""

from typing import Union, Tuple, List, Optional
from ultralytics import YOLO
import numpy as np
import os
import torch
import mlflow
import mlflow.pytorch

from src.config import CLASS_NAMES, MLFLOW_TRACKING_URI

# Check if ONNX Runtime is available
try:
    import onnxruntime as ort
    import cv2
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def load_model(model_path: str = 'yolo11s.pt') -> Union['YOLO', 'ONNXModel']:
    """
    Load YOLOv11s model from checkpoint or ONNX export.
    
    Args:
        model_path: Path to model weights (.pt) or ONNX export (.onnx)
        
    Returns:
        Loaded model instance (YOLO or ONNXModel)
        
    Raises:
        Exception: If model loading fails
    """
    try:
        # Load ONNX model if runtime is available and format matches
        if model_path.endswith('.onnx') and ONNX_AVAILABLE:
            return ONNXModel(model_path)
        else:
            model = YOLO(model_path)
            return model
    except Exception as e:
        raise Exception(f"Model loading failed: {str(e)}")


def train_model(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    resume: bool = False,
    resume_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple:
    """
    Train YOLOv11s model with MLflow tracking.

    Training outputs are saved locally at models/train/weights/best.pt.
    The best checkpoint is also logged as an MLflow artifact.
    Registration to MLflow Model Registry is handled separately by the
    pipeline orchestrator after validation and ONNX export.

    Args:
        data_yaml: Path to dataset configuration YAML file
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Training batch size
        resume: Whether to resume from checkpoint
        resume_path: Path to checkpoint for resuming
        run_id: MLflow run ID for continuing logging

    Returns:
        Tuple of (training results, MLflow run ID)

    Raises:
        Exception: If training fails
    """
    try:
        # Initialize model
        if resume and resume_path:
            model = YOLO(resume_path)
        elif resume:
            model = YOLO('models/train/weights/last.pt')
        else:
            model = YOLO('yolo11s.pt')

        # Setup MLflow tracking
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Resume existing run or create new experiment
        if run_id:
            try:
                run_info = mlflow.get_run(run_id)
                experiment_id = run_info.info.experiment_id
                experiment = mlflow.get_experiment(experiment_id)
                mlflow.set_experiment(experiment.name)
            except Exception as e:
                print(f"Warning: Cannot set experiment from run_id: {str(e)}")
                print("Creating new run instead.")
                run_id = None
        else:
            mlflow.set_experiment("brain_tumor_detection")

        # Enable auto-logging
        mlflow.pytorch.autolog()

        # Train inside an MLflow run
        with mlflow.start_run(run_id=run_id) as run:
            active_run_id = run.info.run_id

            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=False,
                project='models',
                name=None,
                exist_ok=True,
                resume=resume
            )

            # Log key metrics from training results
            try:
                metrics = results.results_dict
                mlflow.log_metrics({
                    "mAP50":     metrics.get("metrics/mAP50(B)", 0),
                    "mAP50_95":  metrics.get("metrics/mAP50-95(B)", 0),
                    "precision": metrics.get("metrics/precision(B)", 0),
                    "recall":    metrics.get("metrics/recall(B)", 0),
                })
            except Exception:
                pass  # metrics logging is best-effort

            # Log best.pt as artifact
            best_pt_path = os.path.join("models", "train", "weights", "best.pt")
            if os.path.exists(best_pt_path):
                mlflow.log_artifact(best_pt_path, artifact_path="weights")
                print(f"✅ Logged artifact: {best_pt_path}")
            else:
                print(f"⚠️  best.pt not found at {best_pt_path}")

        return results, active_run_id

    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")



def predict(model, image: np.ndarray, conf_thres: float = 0.25):
    """
    Run inference on an image.
    
    Args:
        model: Loaded YOLO or ONNXModel instance
        image: Input image (numpy array)
        conf_thres: Confidence threshold for detections
        
    Returns:
        YOLO Results object with detections
        
    Raises:
        Exception: If prediction fails
    """
    try:
        results = model(image, conf=conf_thres)
        return results
    except Exception as e:
        import traceback
        print(f"Detailed prediction error:")
        traceback.print_exc()
        raise Exception(f"Prediction failed: {str(e)}")


# ONNX Model wrapper (only available if onnxruntime is installed)
if ONNX_AVAILABLE:
    class ONNXModel:
        """
        ONNX Runtime wrapper for YOLO model inference.
        
        Provides YOLO-compatible API for ONNX exported models with
        optimized inference using ONNX Runtime.
        """
        
        class Box:
            """Bounding box representation."""
            def __init__(self, xyxy, conf, cls):
                self.xyxy = xyxy
                self.conf = conf
                self.cls = cls
        
        class Boxes:
            """Collection of bounding boxes."""
            def __init__(self):
                self.xyxy = []
                self.conf = []
                self.cls = []
                self._len = 0
                
            def append(self, box):
                """Add a box to the collection."""
                try:
                    # Store box coordinates
                    if isinstance(box.xyxy, np.ndarray):
                        self.xyxy.append(box.xyxy)
                    else:
                        self.xyxy.append(np.array([box.xyxy]))
                        
                    # Store confidence scores
                    if isinstance(box.conf, np.ndarray):
                        self.conf.append(box.conf)
                    else:
                        self.conf.append(np.array([box.conf]))
                        
                    # Store class IDs
                    if isinstance(box.cls, np.ndarray):
                        self.cls.append(box.cls)
                    else:
                        self.cls.append(np.array([box.cls]))
                        
                    self._len += 1
                except Exception as e:
                    print(f"Error adding box: {str(e)}")
            
            def __len__(self):
                return self._len
            
            def __iter__(self):
                """Enable iteration over boxes."""
                class BoxIterator:
                    def __init__(self, boxes):
                        self.boxes = boxes
                        self.index = 0
                        
                    def __iter__(self):
                        return self
                        
                    def __next__(self):
                        while self.index < len(self.boxes):
                            try:
                                # Extract box data
                                xyxy = self.boxes.xyxy[self.index]
                                conf = self.boxes.conf[self.index]
                                cls = self.boxes.cls[self.index]
                                
                                # Convert to numpy arrays if needed
                                if isinstance(xyxy, list):
                                    xyxy = np.array(xyxy)
                                if isinstance(conf, list):
                                    conf = np.array(conf)
                                if isinstance(cls, list):
                                    cls = np.array(cls)
                                    
                                box = ONNXModel.Box(xyxy=xyxy, conf=conf, cls=cls)
                                self.index += 1
                                return box
                            except Exception as e:
                                print(f"Error in iterator at index {self.index}: {str(e)}")
                                self.index += 1
                                continue
                        raise StopIteration
                
                return BoxIterator(self)
        
        class Results:
            """YOLO-compatible results object."""
            def __init__(self, boxes, names):
                self.boxes = boxes
                self.names = names
        
        def __init__(self, model_path: str):
            """
            Initialize ONNX model for inference.
            
            Args:
                model_path: Path to ONNX model file
                
            Raises:
                FileNotFoundError: If model file doesn't exist
            """
            # Verify model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            
            # Create ONNX Runtime session with GPU support
            self.session = ort.InferenceSession(
                model_path, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Parse input shape
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Extract image size from model input shape
            try:
                height = int(self.input_shape[-2]) if isinstance(self.input_shape[-2], (int, float, str)) else 640
                width = int(self.input_shape[-1]) if isinstance(self.input_shape[-1], (int, float, str)) else 640
                self.img_size = (height, width)
            except (ValueError, TypeError, IndexError):
                self.img_size = (640, 640)
                print("Warning: Cannot parse input size from ONNX model, using default (640, 640)")
            
            # Class names from centralized config
            self.names = CLASS_NAMES
        
        def __call__(self, img: np.ndarray, conf: float = 0.25):
            """
            Run inference on image.
            
            Args:
                img: Input image (numpy array)
                conf: Confidence threshold for detections
                
            Returns:
                YOLO-compatible Results object
            """
            # Preprocess image
            input_data = self._preprocess(img)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            # Postprocess results
            results = self._postprocess(outputs, img, conf)
            
            return results
        
        def _preprocess(self, img: np.ndarray) -> np.ndarray:
            """
            Preprocess image for ONNX model inference.
            
            Args:
                img: Input image
                
            Returns:
                Preprocessed image tensor
            """
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Resize to model input size
            try:
                width = int(self.img_size[1])
                height = int(self.img_size[0])
            except (ValueError, TypeError):
                width, height = 640, 640
                
            img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 1]
            img_norm = img_resized.astype(np.float32) / 255.0
            
            # Convert HWC to CHW format
            img_transposed = img_norm.transpose(2, 0, 1)
            
            # Add batch dimension
            img_batch = np.expand_dims(img_transposed, axis=0)
            
            return img_batch
        
        def _postprocess(
            self,
            outputs: List[np.ndarray],
            original_img: np.ndarray,
            conf_threshold: float
        ) -> List:
            """
            Postprocess ONNX model outputs to YOLO format.
            
            Args:
                outputs: Raw ONNX model outputs
                original_img: Original input image
                conf_threshold: Confidence threshold
                
            Returns:
                List of Results objects
            """
            # Extract detections from output
            detections = outputs[0]
            
            # Initialize boxes container
            boxes = self.Boxes()
            
            try:
                # Handle YOLOv8/v11 ONNX output format: (1, 8, 8400)
                if len(detections.shape) == 3:
                    # Transpose from [1, 8, 8400] to [1, 8400, 8] for easier processing
                    if detections.shape[1] <= 8 and detections.shape[2] > 100:
                        detections = detections.transpose(0, 2, 1)
                    
                    # Extract first batch
                    detections = detections[0]  # Shape: [8400, 8]
                    
                    # Parse class scores (assuming first 4 columns are bbox coords)
                    num_classes = min(4, detections.shape[1] - 4)
                    class_scores = detections[:, 4:4+num_classes]
                    max_scores = np.max(class_scores, axis=1)
                    class_ids = np.argmax(class_scores, axis=1)
                    
                    # Filter by confidence threshold
                    mask = max_scores >= conf_threshold
                    
                    if np.any(mask):
                        # Get filtered predictions
                        filtered_boxes = detections[mask, :4]  # [x, y, w, h]
                        filtered_scores = max_scores[mask]
                        filtered_class_ids = class_ids[mask]
                        
                        # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
                        x = filtered_boxes[:, 0]
                        y = filtered_boxes[:, 1]
                        w = filtered_boxes[:, 2]
                        h = filtered_boxes[:, 3]
                        
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        
                        boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)
                        
                        # Scale to original image size
                        orig_h, orig_w = original_img.shape[:2]
                        scale_x = orig_w / self.img_size[1]
                        scale_y = orig_h / self.img_size[0]
                        
                        boxes_xyxy[:, 0] *= scale_x
                        boxes_xyxy[:, 1] *= scale_y
                        boxes_xyxy[:, 2] *= scale_x
                        boxes_xyxy[:, 3] *= scale_y
                        
                        # Filter out "notumor" class (index 2)
                        non_notumor_mask = filtered_class_ids != 2
                        if np.any(non_notumor_mask):
                            boxes_xyxy = boxes_xyxy[non_notumor_mask]
                            filtered_scores = filtered_scores[non_notumor_mask]
                            filtered_class_ids = filtered_class_ids[non_notumor_mask]
                        
                        # Create Box objects
                        for i in range(len(filtered_scores)):
                            try:
                                box = self.Box(
                                    xyxy=np.array([boxes_xyxy[i]]),
                                    conf=np.array([filtered_scores[i]]),
                                    cls=np.array([filtered_class_ids[i]])
                                )
                                boxes.append(box)
                            except Exception as e:
                                print(f"Error creating box {i}: {str(e)}")
                        
                        # Apply Non-Maximum Suppression
                        if len(boxes) > 1:
                            boxes = self._apply_nms(boxes, iou_threshold=0.45)
                else:
                    print(f"Unsupported output format: {detections.shape}")
            except Exception as e:
                print(f"Postprocessing error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Return YOLO-compatible results
            results = [self.Results(boxes, self.names)]
            return results
        
        def _apply_nms(self, boxes, iou_threshold: float = 0.45):
            """
            Apply Non-Maximum Suppression to remove duplicate detections.
            
            Args:
                boxes: Boxes object containing detections
                iou_threshold: IoU threshold for considering boxes as duplicates
                
            Returns:
                Filtered Boxes object after NMS
            """
            if len(boxes) <= 1:
                return boxes
                
            # Initialize filtered boxes
            filtered_boxes = self.Boxes()
            
            # Convert boxes to numpy arrays
            all_xyxy = []
            all_conf = []
            all_cls = []
            
            for box in boxes:
                all_xyxy.append(box.xyxy[0])
                all_conf.append(box.conf[0])
                all_cls.append(box.cls[0])
            
            all_xyxy = np.array(all_xyxy)
            all_conf = np.array(all_conf)
            all_cls = np.array(all_cls)
            
            # Sort by confidence (descending)
            indices = np.argsort(-all_conf)
            
            # Apply NMS algorithm
            keep_indices = []
            while len(indices) > 0:
                # Keep box with highest confidence
                current_idx = indices[0]
                keep_indices.append(current_idx)
                
                if len(indices) == 1:
                    break
                
                # Remove current box from candidates
                indices = indices[1:]
                
                # Calculate IoU with remaining boxes
                current_box = all_xyxy[current_idx]
                remaining_boxes = all_xyxy[indices]
                
                # Calculate box areaselection
                current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                
                # Calculate intersection coordinates
                xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
                yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
                xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
                yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])
                
                # Calculate intersection area
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                intersection = w * h
                
                # Calculate IoU
                union = current_area + remaining_areas - intersection
                iou = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero
                
                # Keep boxes with low IoU or different class
                same_class = all_cls[indices] == all_cls[current_idx]
                low_iou = iou < iou_threshold
                mask = np.logical_or(low_iou, ~same_class)
                indices = indices[mask]
            
            # Create final boxes from kept indices
            for idx in keep_indices:
                box = self.Box(
                    xyxy=np.array([all_xyxy[idx]]),
                    conf=np.array([all_conf[idx]]),
                    cls=np.array([all_cls[idx]])
                )
                filtered_boxes.append(box)
            
            return filtered_boxes
