import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Dynamically add project root and src/feature to sys.path to prevent import errors
project_root = Path(__file__).resolve().parent.parent
src_feature_dir = project_root / "src" / "feature"
for p in [str(project_root), str(src_feature_dir)]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import your PrepareData class (adjust import if needed, e.g., from prepare_data import PrepareData)
from src.feature.prepare_data import PrepareData


@pytest.fixture
def sample_heart_disease_data():
    """Fixture providing a synthetic dataset with target column 'num'."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame(
        {
            "age": np.random.randint(30, 75, size=n_samples),
            "sex": np.random.randint(0, 2, size=n_samples),
            "cp": np.random.randint(0, 4, size=n_samples),
            "trestbps": np.random.randint(90, 200, size=n_samples),
            "chol": np.random.randint(120, 300, size=n_samples),
            "fbs": np.random.randint(0, 2, size=n_samples),
            "restecg": np.random.randint(0, 3, size=n_samples),
            "thalach": np.random.randint(80, 200, size=n_samples),
            "exang": np.random.randint(0, 2, size=n_samples),
            "oldpeak": np.random.uniform(0.0, 6.2, size=n_samples),
            "slope": np.random.randint(0, 3, size=n_samples),
            "ca": np.random.randint(0, 4, size=n_samples),
            "thal": np.random.randint(0, 4, size=n_samples),
            "num": [0, 1] * 25,  # Stratified target binary array
        }
    )


# ------------------------------------------------------------------------------
# 1. Tests for `__init__()`
# ------------------------------------------------------------------------------

@patch("pandas.read_csv")
def test_init_loads_data_from_path(mock_read_csv, sample_heart_disease_data):
    """Test that __init__ reads CSV data from the provided custom path."""
    mock_read_csv.return_value = sample_heart_disease_data
    custom_path = "custom/data/path.csv"

    prep = PrepareData(data_path=custom_path)

    mock_read_csv.assert_called_once_with(custom_path)
    pd.testing.assert_frame_equal(prep.data, sample_heart_disease_data)


# ------------------------------------------------------------------------------
# 2. Tests for `prepare_data()`
# ------------------------------------------------------------------------------

@patch("pandas.read_csv")
def test_prepare_data_train_test_split_shapes(mock_read_csv, sample_heart_disease_data):
    """Test that train_test_split splits data with 80% train and 20% test ratio."""
    mock_read_csv.return_value = sample_heart_disease_data
    prep = PrepareData()

    X_train, X_test, y_train, y_test, preprocessor = prep.prepare_data()

    # 50 total rows -> 40 train rows (80%), 10 test rows (20%)
    assert len(X_train) == 40
    assert len(X_test) == 10
    assert len(y_train) == 40
    assert len(y_test) == 10

    # Ensure target column 'num' is dropped from feature sets
    assert "num" not in X_train.columns
    assert "num" not in X_test.columns


@patch("pandas.read_csv")
def test_prepare_data_stratification(mock_read_csv, sample_heart_disease_data):
    """Test that target distribution 'num' is stratified evenly between train and test."""
    mock_read_csv.return_value = sample_heart_disease_data
    prep = PrepareData()

    _, _, y_train, y_test, _ = prep.prepare_data()

    # Original dataset has 50% class 0 and 50% class 1
    assert (y_train == 0).sum() == 20
    assert (y_train == 1).sum() == 20
    assert (y_test == 0).sum() == 5
    assert (y_test == 1).sum() == 5


@patch("pandas.read_csv")
def test_prepare_data_column_transformer_configuration(mock_read_csv, sample_heart_disease_data):
    """Test that the returned preprocessor is a ColumnTransformer with expected transformer pipelines."""
    mock_read_csv.return_value = sample_heart_disease_data
    prep = PrepareData()

    _, _, _, _, preprocessor = prep.prepare_data()

    # Assert preprocessor type
    assert isinstance(preprocessor, ColumnTransformer)

    # Inspect named transformers
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names
    assert "pass" in transformer_names

    # Check transformed column mappings
    transformers_dict = {name: (trans, cols) for name, trans, cols in preprocessor.transformers}

    assert transformers_dict["num"][1] == ["age", "trestbps", "chol", "thalach", "oldpeak"]
    assert transformers_dict["cat"][1] == ["cp", "restecg", "slope", "thal"]
    assert transformers_dict["pass"][1] == ["sex", "fbs", "exang", "ca"]


@patch("pandas.read_csv")
def test_preprocessor_fit_transform_pipeline(mock_read_csv, sample_heart_disease_data):
    """Test that the returned preprocessor can successfully fit and transform X_train."""
    mock_read_csv.return_value = sample_heart_disease_data
    prep = PrepareData()

    X_train, _, _, _, preprocessor = prep.prepare_data()

    # Execute fit_transform to verify transformer operation on real DataFrame features
    X_train_transformed = preprocessor.fit_transform(X_train)

    assert X_train_transformed.shape[0] == len(X_train)
    # Output columns: 5 num + OneHotEncoded categories + 4 passthrough columns
    assert X_train_transformed.shape[1] > 0