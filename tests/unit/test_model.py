"""
Unit tests for model utilities
"""
from tests.mocks import mock_yolo_class, mock_ort, mock_mlflow

# Now import model utilities and other tools
import os
import pytest
import numpy as np
from src.utils.model import load_model, predict, train_model
from src.config import CLASS_NAMES
from unittest.mock import MagicMock, patch


def test_load_model_pytorch():
    """Test loading PyTorch model"""
    mock_model_instance = MagicMock()
    mock_yolo_class.return_value = mock_model_instance
    
    model = load_model("path/to/model.pt")
    
    mock_yolo_class.assert_called_once_with("path/to/model.pt")
    assert model == mock_model_instance


@patch('src.utils.model.os.path.exists')
def test_load_model_onnx(mock_exists):
    """Test loading ONNX model"""
    mock_exists.return_value = True
    
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    # Configure mock_session inputs/outputs metadata
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_session.get_outputs.return_value = [mock_output]
    
    model = load_model("path/to/model.onnx")
    
    assert model is not None
    mock_ort.InferenceSession.assert_called_once_with(
        "path/to/model.onnx",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )


def test_predict_pytorch():
    """Test prediction with PyTorch model"""
    mock_model = MagicMock()
    mock_results = MagicMock()
    mock_model.return_value = mock_results
    
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = predict(mock_model, image, conf_thres=0.3)
    
    mock_model.assert_called_once_with(image, conf=0.3)
    assert results == mock_results


@patch('src.utils.model.os.path.exists')
def test_predict_onnx(mock_exists):
    """Test prediction with ONNX model"""
    mock_exists.return_value = True
    
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_session.get_outputs.return_value = [mock_output]
    
    # 101 anchors (satisfies > 100 condition to trigger transpose), 8 values each
    dummy_detections = np.zeros((1, 8, 101), dtype=np.float32)
    # Box at index 0
    dummy_detections[0, 0, 0] = 100.0  # x
    dummy_detections[0, 1, 0] = 100.0  # y
    dummy_detections[0, 2, 0] = 50.0   # w
    dummy_detections[0, 3, 0] = 50.0   # h
    dummy_detections[0, 4, 0] = 0.9    # class 0 score (glioma)
    
    mock_session.run.return_value = [dummy_detections]
    
    model = load_model("path/to/model.onnx")
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = predict(model, image, conf_thres=0.25)
    
    assert len(results) == 1
    res = results[0]
    assert res.names == CLASS_NAMES
    
    boxes = res.boxes
    assert len(boxes) == 1
    box = next(iter(boxes))
    # Coordinates: [75.0, 75.0, 125.0, 125.0]
    np.testing.assert_allclose(box.xyxy[0], [75.0, 75.0, 125.0, 125.0])
    np.testing.assert_allclose(box.conf[0], 0.9)
    assert box.cls[0] == 0


@patch('src.utils.model.os.path.exists')
def test_predict_onnx_with_nms(mock_exists):
    """Test prediction with ONNX model NMS"""
    mock_exists.return_value = True
    
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_session.get_outputs.return_value = [mock_output]
    
    # 101 anchors, 8 values each
    dummy_detections = np.zeros((1, 8, 101), dtype=np.float32)
    
    # Box 1
    dummy_detections[0, 0, 0] = 100.0
    dummy_detections[0, 1, 0] = 100.0
    dummy_detections[0, 2, 0] = 50.0
    dummy_detections[0, 3, 0] = 50.0
    dummy_detections[0, 4, 0] = 0.9
    
    # Box 2 (overlapping, should be suppressed)
    dummy_detections[0, 0, 1] = 102.0
    dummy_detections[0, 1, 1] = 100.0
    dummy_detections[0, 2, 1] = 50.0
    dummy_detections[0, 3, 1] = 50.0
    dummy_detections[0, 4, 1] = 0.8
    
    mock_session.run.return_value = [dummy_detections]
    
    model = load_model("path/to/model.onnx")
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = predict(model, image, conf_thres=0.25)
    
    assert len(results) == 1
    res = results[0]
    boxes = res.boxes
    assert len(boxes) == 1
    box = next(iter(boxes))
    np.testing.assert_allclose(box.xyxy[0], [75.0, 75.0, 125.0, 125.0])
    np.testing.assert_allclose(box.conf[0], 0.9)


def test_predict_output_format():
    """Test prediction output format mapping and structures"""
    # Verify we can construct and inspect the Box, Boxes, and Results output format
    # directly using ONNXModel classes
    from src.utils.model import ONNXModel
    
    # Test Boxes collection class
    boxes = ONNXModel.Boxes()
    assert len(boxes) == 0
    
    box1 = ONNXModel.Box(xyxy=np.array([[10, 20, 30, 40]]), conf=np.array([0.85]), cls=np.array([1]))
    box2 = ONNXModel.Box(xyxy=[50, 60, 70, 80], conf=0.9, cls=3)
    
    boxes.append(box1)
    boxes.append(box2)
    
    assert len(boxes) == 2
    
    # Iterate over boxes and verify type/data
    iterated = list(boxes)
    assert len(iterated) == 2
    
    assert isinstance(iterated[0].xyxy, np.ndarray)
    assert isinstance(iterated[0].conf, np.ndarray)
    assert isinstance(iterated[0].cls, np.ndarray)
    
    np.testing.assert_allclose(iterated[0].xyxy[0], [10, 20, 30, 40])
    np.testing.assert_allclose(iterated[0].conf[0], 0.85)
    assert iterated[0].cls[0] == 1
    
    np.testing.assert_allclose(iterated[1].xyxy[0], [50, 60, 70, 80])
    np.testing.assert_allclose(iterated[1].conf[0], 0.9)
    assert iterated[1].cls[0] == 3


@patch('src.utils.model.os.path.exists')
def test_train_model(mock_exists):
    """Test training model pipeline and mlflow logging"""
    mock_exists.return_value = True
    
    mock_model_instance = MagicMock()
    mock_yolo_class.return_value = mock_model_instance
    
    mock_results = MagicMock()
    mock_results.results_dict = {
        "metrics/mAP50(B)": 0.85,
        "metrics/mAP50-95(B)": 0.65,
        "metrics/precision(B)": 0.88,
        "metrics/recall(B)": 0.82
    }
    mock_model_instance.train.return_value = mock_results
    
    # Mock MLflow run context
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    
    results, run_id = train_model(
        data_yaml="configs/data.yaml",
        epochs=10,
        imgsz=640,
        batch_size=8
    )
    
    mock_yolo_class.assert_called_once_with('yolo11s.pt')
    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("brain_tumor_detection")
    mock_mlflow.pytorch.autolog.assert_called_once()
    
    mock_model_instance.train.assert_called_once_with(
        data="configs/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        device='cpu',
        verbose=False,
        project='models',
        name=None,
        exist_ok=True,
        resume=False
    )
    
    mock_mlflow.log_metrics.assert_called_once_with({
        "mAP50": 0.85,
        "mAP50_95": 0.65,
        "precision": 0.88,
        "recall": 0.82
    })
    mock_mlflow.log_artifact.assert_called_once_with(
        os.path.join("models", "train", "weights", "best.pt"),
        artifact_path="weights"
    )
    
    assert results == mock_results
    assert run_id == "test_run_123"


def test_predict_invalid_inputs_pytorch_exception():
    """Test that predict raises Exception when PyTorch model fails on invalid input"""
    mock_model = MagicMock()
    mock_model.side_effect = Exception("Invalid input format")
    
    with pytest.raises(Exception, match="Prediction failed: Invalid input format"):
        predict(mock_model, np.zeros((640, 640, 3), dtype=np.uint8))


@patch('src.utils.model.os.path.exists')
def test_predict_onnx_empty_image(mock_exists):
    """Test ONNX model prediction on empty/zero-size image"""
    mock_exists.return_value = True
    
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    # Configure mock_session inputs/outputs metadata
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_session.get_outputs.return_value = [mock_output]
    
    model = load_model("path/to/model.onnx")
    
    # Empty image (height = 0 or width = 0)
    empty_image = np.zeros((0, 640, 3), dtype=np.uint8)
    
    with pytest.raises(Exception, match="Prediction failed"):
        predict(model, empty_image)


@patch('src.utils.model.os.path.exists')
def test_predict_onnx_unsupported_format(mock_exists):
    """Test ONNX model prediction on unsupported formats (e.g. string or None)"""
    mock_exists.return_value = True
    
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_session.get_outputs.return_value = [mock_output]
    
    model = load_model("path/to/model.onnx")
    
    # Case A: String input instead of np.ndarray
    with pytest.raises(Exception, match="Prediction failed"):
        predict(model, "invalid-image-format-string")
        
    # Case B: None input
    with pytest.raises(Exception, match="Prediction failed"):
        predict(model, None)


@patch('src.utils.model.os.path.exists')
def test_predict_onnx_incorrect_dimensions(mock_exists):
    """Test ONNX model prediction on images with incorrect dimensions"""
    mock_exists.return_value = True
    
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_session.get_outputs.return_value = [mock_output]
    
    model = load_model("path/to/model.onnx")
    
    # Incorrect dimension: 1D array instead of HWC (3D)
    incorrect_dim_image = np.zeros((1000,), dtype=np.uint8)
    
    with pytest.raises(Exception, match="Prediction failed"):
        predict(model, incorrect_dim_image)
