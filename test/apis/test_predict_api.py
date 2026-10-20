import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.apis.predict_api import app, PatientFeatures

# Valid payload fixture matching PatientFeatures model
@pytest.fixture
def sample_payload():
    return {
        "age": 63.0,
        "sex": 1,
        "cp": 3,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }


# Mock MLflow model fixture
@pytest.fixture
def mock_mlflow_model():
    mock_model = MagicMock()
    # Predict returns array [1]
    mock_model.predict.return_value = np.array([1])
    # Mock predict_proba method
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return mock_model


# ------------------------------------------------------------------------------
# 1. Health Check & Lifespan Tests
# ------------------------------------------------------------------------------

def test_health_check_when_model_not_loaded():
    """Test /health endpoint when model is None."""
    with patch("src.apis.predict_api.model", None):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy", "model_loaded": False}


def test_health_check_when_model_loaded(mock_mlflow_model):
    """Test /health endpoint when model is loaded."""
    with patch("src.apis.predict_api.model", mock_mlflow_model):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy", "model_loaded": True}


# ------------------------------------------------------------------------------
# 2. Prediction Endpoint (/predict) Tests
# ------------------------------------------------------------------------------

def test_predict_model_not_loaded(sample_payload):
    """Test /predict returns HTTP 503 Service Unavailable when model is None."""
    with patch("src.apis.predict_api.model", None):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 503
            assert response.json()["detail"] == "Model is not loaded"


def test_predict_success(sample_payload, mock_mlflow_model):
    """Test successful prediction with class and probabilities."""
    with patch("src.apis.predict_api.model", mock_mlflow_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 200

            data = response.json()
            assert "predicted_class" in data
            assert data["predicted_class"] == 1
            assert "probabilities" in data
            assert data["probabilities"] == {"class_0": 0.2, "class_1": 0.8}


def test_predict_without_proba(sample_payload):
    """Test prediction for models without predict_proba method."""
    mock_model_no_proba = MagicMock(spec=["predict"])  # Has no predict_proba
    mock_model_no_proba.predict.return_value = np.array([0])

    with patch("src.apis.predict_api.model", mock_model_no_proba):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["predicted_class"] == 0
            assert "probabilities" not in data


def test_predict_model_exception(sample_payload, mock_mlflow_model):
    """Test /predict handles internal model runtime exceptions gracefully."""
    mock_mlflow_model.predict.side_effect = Exception("Model runtime error")

    with patch("src.apis.predict_api.model", mock_mlflow_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 400
            assert "Prediction error: Model runtime error" in response.json()["detail"]


# ------------------------------------------------------------------------------
# 3. Input Data Validation Tests
# ------------------------------------------------------------------------------

def test_predict_invalid_schema_missing_field(sample_payload):
    """Test 422 Unprocessable Entity when a required feature field is missing."""
    del sample_payload["age"]  # Missing required field

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_payload)
        assert response.status_code == 422


def test_predict_invalid_data_type(sample_payload):
    """Test 422 Unprocessable Entity when invalid data types are sent."""
    sample_payload["age"] = "not_a_number"

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_payload)
        assert response.status_code == 422


# ------------------------------------------------------------------------------
# 4. Prometheus Metrics Endpoint Test
# ------------------------------------------------------------------------------

def test_metrics_endpoint_exposed():
    """Test that /metrics endpoint is exposed by Prometheus instrumentator."""
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text