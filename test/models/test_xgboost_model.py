import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

# Automatically resolve project root and src/ directories to prevent ModuleNotFoundError
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"
for p in [str(project_root), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.xgboost_model import XGBoostPipeline


@pytest.fixture
def synthetic_multiclass_data_splits():
    """Fixture providing synthetic multiclass data matching PrepareData output structure."""
    np.random.seed(42)
    n_train, n_test = 40, 10

    cols = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
        "cp", "restecg", "slope", "thal", "sex", "fbs", "exang", "ca"
    ]

    X_train = pd.DataFrame(np.random.randn(n_train, len(cols)), columns=cols)
    X_test = pd.DataFrame(np.random.randn(n_test, len(cols)), columns=cols)

    # 3 distinct target classes (0, 1, 2) required for multi:softprob & roc_auc_ovr
    y_train = np.array([0, 1, 2, 0] * 10)
    y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    preprocessor = MagicMock()

    return X_train, X_test, y_train, y_test, preprocessor


# ------------------------------------------------------------------------------
# 1. Tests for Initialization & Model Building
# ------------------------------------------------------------------------------

def test_init_default_parameters():
    """Test default constructor initialization."""
    pipeline_obj = XGBoostPipeline()
    assert pipeline_obj.use_sample_weights is True
    assert pipeline_obj.model is None


def test_build_model_configuration():
    """Test build_model creates an XGBClassifier configured for multiclass."""
    pipeline_obj = XGBoostPipeline()
    model = pipeline_obj.build_model()

    assert isinstance(model, XGBClassifier)
    assert model.n_estimators == 150
    assert model.max_depth == 4
    assert model.learning_rate == 0.05
    assert model.objective == "multi:softprob"
    assert model.eval_metric == "mlogloss"


# ------------------------------------------------------------------------------
# 2. Integration & Orchestration Tests for `train_and_evaluate()`
# ------------------------------------------------------------------------------

# PATCH WHERE CONSUMED: src.models.xgboost_model.PrepareData
@patch("src.models.xgboost_model.PrepareData")
def test_train_and_evaluate_with_sample_weights(
        mock_prepare_data_cls, synthetic_multiclass_data_splits
):
    """Test train_and_evaluate executes with sample weights enabled."""
    mock_prep_instance = MagicMock()
    mock_prep_instance.prepare_data.return_value = synthetic_multiclass_data_splits
    mock_prepare_data_cls.return_value = mock_prep_instance

    pipeline_obj = XGBoostPipeline(use_sample_weights=True)
    trained_model = pipeline_obj.train_and_evaluate()

    mock_prepare_data_cls.assert_called_once()
    mock_prep_instance.prepare_data.assert_called_once()

    assert isinstance(trained_model, XGBClassifier)
    assert pipeline_obj.model == trained_model


@patch("src.models.xgboost_model.PrepareData")
def test_train_and_evaluate_without_sample_weights(
        mock_prepare_data_cls, synthetic_multiclass_data_splits
):
    """Test train_and_evaluate executes with sample weights disabled."""
    mock_prep_instance = MagicMock()
    mock_prep_instance.prepare_data.return_value = synthetic_multiclass_data_splits
    mock_prepare_data_cls.return_value = mock_prep_instance

    pipeline_obj = XGBoostPipeline(use_sample_weights=False)
    trained_model = pipeline_obj.train_and_evaluate()

    assert isinstance(trained_model, XGBClassifier)


@patch("src.models.xgboost_model.PrepareData")
def test_trained_model_predictions_and_probabilities(
        mock_prepare_data_cls, synthetic_multiclass_data_splits
):
    """Test that the trained XGBoost model outputs predictions and probability vectors."""
    mock_prep_instance = MagicMock()
    mock_prep_instance.prepare_data.return_value = synthetic_multiclass_data_splits
    mock_prepare_data_cls.return_value = mock_prep_instance

    _, X_test, _, _, _ = synthetic_multiclass_data_splits

    pipeline_obj = XGBoostPipeline(use_sample_weights=True)
    trained_model = pipeline_obj.train_and_evaluate()

    # Predict discrete class labels
    predictions = trained_model.predict(X_test)
    assert len(predictions) == len(X_test)

    # Predict probabilities (shape: [n_samples, 3] for 3 classes)
    probabilities = trained_model.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 3)
    np.testing.assert_array_almost_equal(probabilities.sum(axis=1), np.ones(len(X_test)))
