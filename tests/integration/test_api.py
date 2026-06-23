"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.servering.api import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_source" in data


def test_predict_endpoint_no_file():
    """Test predict endpoint without file"""
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error


# TODO: Add test with actual image file
# def test_predict_endpoint_with_file():
#     """Test predict endpoint with valid image"""
#     with open("tests/fixtures/sample_mri.jpg", "rb") as f:
#         response = client.post(
#             "/predict",
#             files={"file": ("test.jpg", f, "image/jpeg")},
#             params={"confidence": 0.5}
#         )
#     assert response.status_code == 200
#     data = response.json()
#     assert "predictions" in data
