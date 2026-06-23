"""
Integration tests for YOLOWrapper and config integration.
"""
from tests.mocks import mock_yolo_class, mock_ort

import os
import pytest
import numpy as np
from src.utils.yolo_wrapper import YOLOWrapper
from src.config import CLASS_NAMES, DEFAULT_CONFIDENCE
from unittest.mock import MagicMock, patch


class MockContext:
    """Mock MLflow context for loading model artifacts."""
    def __init__(self, pt_path, onnx_path):
        self.artifacts = {
            "best_pt": pt_path,
            "best_onnx": onnx_path
        }


@patch('src.utils.model.os.path.exists')
def test_yolo_wrapper_initialization_and_config(mock_exists):
    """
    Test that YOLOWrapper initializes correctly and integrates configurations from config.py.
    """
    mock_exists.return_value = True
    
    # 1. Setup mock models
    mock_pt_model = MagicMock()
    mock_yolo_class.return_value = mock_pt_model
    
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    
    # Configure mock_session inputs/outputs metadata for ONNX
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    mock_session.get_inputs.return_value = [mock_input]
    
    mock_output = MagicMock()
    mock_output.name = "output0"
    mock_session.get_outputs.return_value = [mock_output]
    
    # 2. Instantiate wrapper and load context
    wrapper = YOLOWrapper()
    context = MockContext(pt_path="path/to/best.pt", onnx_path="path/to/best.onnx")
    
    wrapper.load_context(context)
    
    # Assert PyTorch model loaded with the correct path from context
    mock_yolo_class.assert_called_once_with("path/to/best.pt")
    assert wrapper.pt_model == mock_pt_model
    
    # Assert ONNX model is not loaded initially (lazy loading verification)
    assert wrapper._onnx_model is None
    assert wrapper.onnx_available is True
    
    # 3. Trigger lazy loading of ONNX model
    onnx_model = wrapper.onnx_model
    assert onnx_model is not None
    assert wrapper._onnx_model == onnx_model
    
    # Assert that ONNXModel has loaded class names correctly from src.config.CLASS_NAMES
    assert onnx_model.names == CLASS_NAMES
    
    # 4. Verify predict flow and configuration parameters
    # Dummy image: 640x640x3
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run PyTorch prediction (default)
    mock_pt_results = MagicMock()
    mock_pt_model.return_value = mock_pt_results
    
    # We test that the default confidence threshold matches DEFAULT_CONFIDENCE from config.py
    results_pt = wrapper.predict(context=None, model_input=image)
    mock_pt_model.assert_called_once_with(image, conf=DEFAULT_CONFIDENCE)
    assert results_pt == mock_pt_results
    
    # Run ONNX prediction
    dummy_detections = np.zeros((1, 8, 101), dtype=np.float32)
    # Box at index 0, class 1 (meningioma)
    dummy_detections[0, 0, 0] = 100.0
    dummy_detections[0, 1, 0] = 100.0
    dummy_detections[0, 2, 0] = 50.0
    dummy_detections[0, 3, 0] = 50.0
    dummy_detections[0, 4, 0] = 0.1  # class 0 score
    dummy_detections[0, 5, 0] = 0.9  # class 1 score (strictly greater, so argmax is 1)
    mock_session.run.return_value = [dummy_detections]
    
    results_onnx = wrapper.predict(context=None, model_input=image, params={"use_onnx": True})
    
    assert len(results_onnx) == 1
    res = results_onnx[0]
    # Check that predictions map to configured CLASS_NAMES
    assert res.names == CLASS_NAMES
    assert len(res.boxes) == 1
    box = next(iter(res.boxes))
    assert box.cls[0] == 1  # meningioma
