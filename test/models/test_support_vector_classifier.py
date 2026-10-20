import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Dynamically resolve sys.path for test execution across different environments
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"
for p in [str(project_root), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.support_vector_classifier import SVClassification


@pytest.fixture
def synthetic_data_splits():
    """Fixture generating synthetic feature splits and preprocessor matching PrepareData outputs."""
    np.random.seed(42)
    n_train, n_test = 40, 10

    cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "cp", "restecg", "slope", "thal", "sex", "fbs", "exang",
            "ca"]

    X_train = pd.DataFrame(np.random.randn(n_train, len(cols)), columns=cols)
    X_test = pd.DataFrame(np.random.randn(n_test, len(cols)), columns=cols)

    # Stratified target arrays matching binary classification
    y_train = np.array([0, 1] * 20)
    y_test = np.array([0, 1] * 5)

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
    svc_obj = SVClassification()
    assert svc_obj.C == 1.0
    assert svc_obj.kernel == "rbf"
    assert svc_obj.gamma == "scale"
    assert svc_obj.pipeline is None


def test_build_model_configuration():
    """Test _build_model wraps SVC inside CalibratedClassifierCV with specified parameters."""
    svc_obj = SVClassification(C=0.5, kernel="linear", gamma="auto")
    calibrated_model = svc_obj._build_model()

    assert isinstance(calibrated_model, CalibratedClassifierCV)
    assert calibrated_model.method == "sigmoid"
    assert calibrated_model.ensemble is False

    base_svc = calibrated_model.estimator
    assert isinstance(base_svc, SVC)
    assert base_svc.C == 0.5
    assert base_svc.kernel == "linear"
    assert base_svc.gamma == "auto"
    assert base_svc.class_weight == "balanced"
    assert base_svc.random_state == 42

