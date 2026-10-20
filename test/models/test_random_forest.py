import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Dynamically resolve sys.path for test execution across different environments
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
for p in [str(project_root), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import your model pipeline (adjust import path if needed, e.g., from src.models.random_forest_pipeline import RandomForestPipeline)
from src.models.random_forest import RandomForestPipeline


@pytest.fixture
def synthetic_data_splits():
    """Fixture generating synthetic feature splits and preprocessor matching PrepareData outputs."""
    np.random.seed(42)
    n_train, n_test = 40, 10

    # Create dummy raw DataFrames matching expected feature columns
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "cp", "restecg", "slope", "thal", "sex", "fbs", "exang",
            "ca"]

    X_train = pd.DataFrame(np.random.randn(n_train, len(cols)), columns=cols)
    X_test = pd.DataFrame(np.random.randn(n_test, len(cols)), columns=cols)

    # Stratified target arrays
    y_train = np.array([0, 1] * 20)
    y_test = np.array([0, 1] * 5)

    # ColumnTransformer matching PrepareData structure
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["age", "trestbps", "chol", "thalach", "oldpeak"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cp", "restecg", "slope", "thal"]),
            ("pass", "passthrough", ["sex", "fbs", "exang", "ca"]),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


# ------------------------------------------------------------------------------
# 1. Tests for Initialization & Model Building
# ------------------------------------------------------------------------------

def test_init_default_hyperparameters():
    """Test default hyperparameter assignment in constructor."""
    model_obj = RandomForestPipeline()
    assert model_obj.n_estimators == 200
    assert model_obj.max_depth == 5
    assert model_obj.min_samples_leaf == 3
    assert model_obj.pipeline is None


def test_build_model_configuration():
    """Test _build_model creates a RandomForestClassifier with expected parameters."""
    model_obj = RandomForestPipeline(n_estimators=100, max_depth=3, min_samples_leaf=2)
    clf = model_obj._build_model()

    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 100
    assert clf.max_depth == 3
    assert clf.min_samples_leaf == 2
    assert clf.class_weight == "balanced_subsample"
    assert clf.random_state == 42


# ------------------------------------------------------------------------------
# 2. Integration & Orchestration Tests for `train_and_evaluate()`
# ------------------------------------------------------------------------------

# Note: Patch PrepareData where it is used in your pipeline module
@patch("src.models.random_forest.PrepareData")
def test_train_and_evaluate_returns_fitted_pipeline(
        mock_prepare_data_cls, synthetic_data_splits
):
    """Test full training workflow returns a fitted Pipeline instance."""
    # Setup PrepareData mock
    mock_prep_instance = MagicMock()
    mock_prep_instance.prepare_data.return_value = synthetic_data_splits
    mock_prepare_data_cls.return_value = mock_prep_instance

    # Initialize and train pipeline
    rf_pipeline = RandomForestPipeline(n_estimators=10, max_depth=3, min_samples_leaf=1)
    trained_pipeline = rf_pipeline.train_and_evaluate()

    # Assertions
    mock_prepare_data_cls.assert_called_once()
    mock_prep_instance.prepare_data.assert_called_once()

    assert isinstance(trained_pipeline, Pipeline)
    assert "preprocessor" in trained_pipeline.named_steps
    assert "classifier" in trained_pipeline.named_steps
    assert rf_pipeline.pipeline == trained_pipeline


@patch("src.models.random_forest.PrepareData")
def test_trained_pipeline_can_predict(mock_prepare_data_cls, synthetic_data_splits):
    """Test that the returned pipeline can execute predictions on new input features."""
    mock_prep_instance = MagicMock()
    mock_prep_instance.prepare_data.return_value = synthetic_data_splits
    mock_prepare_data_cls.return_value = mock_prep_instance

    X_train, X_test, _, _, _ = synthetic_data_splits

    rf_pipeline = RandomForestPipeline(n_estimators=10, max_depth=3)
    trained_pipeline = rf_pipeline.train_and_evaluate()

    # Predict using trained pipeline
    predictions = trained_pipeline.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})


@patch("src.models.random_forest.PrepareData")
def test_trained_pipeline_has_predict_proba(mock_prepare_data_cls, synthetic_data_splits):
    """Test that the fitted pipeline exposes predict_proba for probabilities output."""
    mock_prep_instance = MagicMock()
    mock_prep_instance.prepare_data.return_value = synthetic_data_splits
    mock_prepare_data_cls.return_value = mock_prep_instance

    _, X_test, _, _, _ = synthetic_data_splits

    rf_pipeline = RandomForestPipeline(n_estimators=10, max_depth=3)
    trained_pipeline = rf_pipeline.train_and_evaluate()

    probabilities = trained_pipeline.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 2)
    # Check probabilities sum to 1
    np.testing.assert_array_almost_equal(probabilities.sum(axis=1), np.ones(len(X_test)))