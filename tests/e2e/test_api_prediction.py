"""
End-to-end tests for FastAPI prediction endpoint.
"""
from tests.mocks import mock_yolo_class

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from src.config import CLASS_NAMES


@pytest.mark.parametrize("class_id,expected_category", [
    (0, "glioma"),
    (1, "meningioma"),
    (2, "notumor"),
    (3, "pituitary"),
])
def test_predict_endpoint_e2e(class_id, expected_category):
    """
    E2E test for the /predict endpoint.
    Verifies that the API receives an image, performs brain tumor prediction,
    and returns a 200 response with correct classification category mapping.
    """
    # 1. Configure the mock YOLO model first
    mock_pt_model = MagicMock()
    mock_yolo_class.return_value = mock_pt_model
    
    # Mock YOLO Results structure
    mock_result = MagicMock()
    mock_result.names = CLASS_NAMES
    
    # Bounding box mock
    mock_box = MagicMock()
    mock_box.conf = [0.85]
    mock_box.cls = [class_id]
    mock_box.xyxy = [np.array([10.0, 20.0, 30.0, 40.0])]
    mock_result.boxes = [mock_box]
    
    mock_pt_model.return_value = [mock_result]

    # 2. Start TestClient (triggers startup event)
    from src.servering.api import app
    with TestClient(app) as client:
        # 3. Direct inject mocks into api module after startup finished
        import src.servering.api as api
        api.model = mock_pt_model
        api.model_source = "Local/fallback"
        api.model_type = "PyTorch"
        
        # Clear the prediction service cache
        api.prediction_service._cache.clear()
        
        # Create a dummy image
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        # Send POST request
        response = client.post(
            "/predict",
            files={"file": ("mri_sample.jpg", img_bytes, "image/jpeg")},
            data={"use_onnx": False, "confidence": 0.5}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "message" in data
        assert data["message"] == "Prediction completed successfully"
        assert "processing_time" in data
        assert "request_id" in data
        
        predictions = data["predictions"]
        assert len(predictions) == 1
        pred = predictions[0]
        
        # Check correct classification category mapping
        assert pred["class"] == expected_category
        assert pred["confidence"] == 0.85
        assert pred["bbox"] == [10.0, 20.0, 30.0, 40.0]
