import sys
from unittest.mock import MagicMock

# Create reusable mocks
mock_yolo_class = MagicMock()
mock_ort = MagicMock()
mock_torch = MagicMock()
mock_mlflow = MagicMock()
mock_mlflow_pytorch = MagicMock()
mock_mlflow_tracking = MagicMock()

# Setup default returns / structures
mock_torch.cuda.is_available.return_value = False
mock_mlflow.pytorch = mock_mlflow_pytorch
mock_mlflow.tracking = mock_mlflow_tracking

# Real Python base class so that subclassing doesn't turn the subclass into a MagicMock
class DummyPythonModel:
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input, params=None):
        pass

class DummyMlflowPyfunc:
    PythonModel = DummyPythonModel

# Let's create dummy classes for the modules to hold the attributes deterministically
class DummyUltralytics:
    YOLO = mock_yolo_class

class DummyOnnxruntime:
    InferenceSession = mock_ort.InferenceSession

class DummyTorch:
    cuda = mock_torch.cuda

class DummyMlflow:
    pytorch = mock_mlflow_pytorch
    pyfunc = DummyMlflowPyfunc()
    tracking = mock_mlflow_tracking
    
    def set_tracking_uri(self, *args, **kwargs):
        return mock_mlflow.set_tracking_uri(*args, **kwargs)
        
    def set_experiment(self, *args, **kwargs):
        return mock_mlflow.set_experiment(*args, **kwargs)
        
    def start_run(self, *args, **kwargs):
        return mock_mlflow.start_run(*args, **kwargs)
        
    def log_params(self, *args, **kwargs):
        return mock_mlflow.log_params(*args, **kwargs)
        
    def log_metrics(self, *args, **kwargs):
        return mock_mlflow.log_metrics(*args, **kwargs)
        
    def log_artifact(self, *args, **kwargs):
        return mock_mlflow.log_artifact(*args, **kwargs)
        
    def get_run(self, *args, **kwargs):
        return mock_mlflow.get_run(*args, **kwargs)
        
    def get_experiment(self, *args, **kwargs):
        return mock_mlflow.get_experiment(*args, **kwargs)

# Populate sys.modules
sys.modules['ultralytics'] = DummyUltralytics()
sys.modules['onnxruntime'] = DummyOnnxruntime()
sys.modules['torch'] = DummyTorch()
sys.modules['mlflow'] = DummyMlflow()
sys.modules['mlflow.pytorch'] = mock_mlflow_pytorch
sys.modules['mlflow.pyfunc'] = DummyMlflowPyfunc()
sys.modules['mlflow.tracking'] = mock_mlflow_tracking
