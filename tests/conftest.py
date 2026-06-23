"""
Test configuration and fixtures
"""
# Import mocks first to ensure sys.modules is pre-populated
import tests.mocks
import pytest


@pytest.fixture(autouse=True)
def reset_shared_mocks():
    """Autouse fixture to reset the shared mocks before each test run."""
    from tests.mocks import mock_yolo_class, mock_ort, mock_mlflow, mock_torch
    mock_yolo_class.reset_mock()
    mock_ort.reset_mock()
    mock_mlflow.reset_mock()
    mock_torch.reset_mock()


@pytest.fixture
def sample_image_path():
    """Fixture providing path to sample test image"""
    return "tests/fixtures/sample_mri.jpg"


@pytest.fixture
def api_client():
    """Fixture providing FastAPI test client"""
    from fastapi.testclient import TestClient
    from src.servering.api import app
    return TestClient(app)
