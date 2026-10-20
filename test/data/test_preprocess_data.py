import sys
from pathlib import Path

# Automatically append project root and src/data to sys.path to prevent import errors
project_root = Path(__file__).resolve().parent.parent.parent
src_data_dir = project_root / "src" / "data"
for p in [str(project_root), str(src_data_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess_data import PreprocessData


@pytest.fixture
def sample_raw_dataframe():
    """Fixture providing a sample DataFrame with missing values in 'ca' and 'thal'."""
    return pd.DataFrame(
        {
            "age": [63.0, 37.0, 41.0, 56.0, 57.0],
            "sex": [1, 1, 0, 1, 0],
            "ca": [0.0, 1.0, np.nan, 0.0, np.nan],  # Mode is 0.0
            "thal": [1.0, 2.0, 2.0, np.nan, 2.0],  # Mode is 2.0
        }
    )


@pytest.fixture
def preprocess_instance():
    """Fixture providing a fresh instance of PreprocessData."""
    return PreprocessData()


# ------------------------------------------------------------------------------
# 1. Tests for `missing_values()`
# ------------------------------------------------------------------------------

def test_missing_values_imputation(preprocess_instance, sample_raw_dataframe):
    """Test that missing values in 'ca' and 'thal' are correctly imputed with their mode."""
    assert sample_raw_dataframe["ca"].isnull().sum() == 2
    assert sample_raw_dataframe["thal"].isnull().sum() == 1

    clean_df = preprocess_instance.missing_values(sample_raw_dataframe)

    assert clean_df["ca"].isnull().sum() == 0
    assert clean_df["thal"].isnull().sum() == 0

    assert clean_df["ca"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert clean_df["thal"].tolist() == [1.0, 2.0, 2.0, 2.0, 2.0]


# ------------------------------------------------------------------------------
# 2. Tests for `save_data()`
# ------------------------------------------------------------------------------

@patch("pathlib.Path.mkdir")
def test_save_data_creates_directory_and_saves_csv(mock_mkdir, preprocess_instance):
    """Test that save_data creates output directories and calls to_csv on self.data."""
    preprocess_instance.data = MagicMock()
    test_output_path = "dummy/path/output.csv"

    preprocess_instance.save_data(test_output_path)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    preprocess_instance.data.to_csv.assert_called_once_with(Path(test_output_path), index=False)


# ------------------------------------------------------------------------------
# 3. Integration Tests for `data_preprocess()`
# ------------------------------------------------------------------------------

# Note: Patching 'src.data.preprocess_data.ValidateData' where it is consumed
@patch("src.data.preprocess_data.ValidateData")
@patch.object(PreprocessData, "save_data")
def test_data_preprocess_orchestration(
    mock_save_data, mock_validate_data_cls, preprocess_instance, sample_raw_dataframe
):
    """Test full orchestration: loading dataset via ValidateData, imputing missing values, and triggering save."""
    mock_validate_instance = MagicMock()
    mock_validate_instance.load_dataset.return_value = sample_raw_dataframe
    mock_validate_data_cls.return_value = mock_validate_instance

    result_df = preprocess_instance.data_preprocess()

    mock_validate_data_cls.assert_called_once()
    mock_validate_instance.load_dataset.assert_called_once()

    assert preprocess_instance.data is not None
    assert preprocess_instance.data["ca"].isnull().sum() == 0

    mock_save_data.assert_called_once_with(preprocess_instance.output_path)
    pd.testing.assert_frame_equal(result_df, preprocess_instance.data)