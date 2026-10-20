import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

# Dynamically add project root and src/data to sys.path to prevent import errors
project_root = Path(__file__).resolve().parent.parent
src_data_dir = project_root / "src" / "data"
for path in [str(project_root), str(src_data_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import ValidateData (adjust import module string if needed, e.g., from src.data.validate_data import ValidateData)
from src.data.validate_data import ValidateData


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame."""
    return pd.DataFrame(
        {
            "age": [63.0, 37.0, 41.0],
            "sex": [1, 1, 0],
            "cp": [3, 2, 1],
            "trestbps": [145.0, 130.0, 130.0],
            "chol": [233.0, 250.0, 204.0],
        }
    )


@pytest.fixture
def validator_instance():
    """Fixture providing a fresh instance of ValidateData."""
    return ValidateData()


# ------------------------------------------------------------------------------
# 1. Tests for `load_dataset()`
# ------------------------------------------------------------------------------

@patch("pandas.read_csv")
def test_load_dataset_success(mock_read_csv, validator_instance, sample_dataframe):
    """Test successful reading of CSV file using pandas.read_csv."""
    mock_read_csv.return_value = sample_dataframe

    df = validator_instance.load_dataset()

    # Verify pandas.read_csv was called with correct default path
    mock_read_csv.assert_called_once_with(validator_instance.file_path)

    # Verify self.data is populated correctly
    assert validator_instance.data is not None
    pd.testing.assert_frame_equal(df, sample_dataframe)
    pd.testing.assert_frame_equal(validator_instance.data, sample_dataframe)


@patch("pandas.read_csv")
@patch("builtins.print")
def test_load_dataset_failure_handles_exception(mock_print, mock_read_csv, validator_instance):
    """Test that file loading exceptions (e.g., FileNotFoundError) are caught and printed."""
    error_exception = FileNotFoundError("No such file or directory")
    mock_read_csv.side_effect = error_exception

    df = validator_instance.load_dataset()

    # Verify return value is None on failure
    assert df is None
    assert validator_instance.data is None

    # Verify print statement was executed with error detail
    mock_print.assert_called_with("Failed to load data ", error_exception)


# ------------------------------------------------------------------------------
# 2. Integration Tests for `validate_data()`
# ------------------------------------------------------------------------------

@patch.object(ValidateData, "load_dataset")
def test_validate_data_calls_load_dataset_and_inspects(mock_load_dataset, validator_instance, sample_dataframe):
    """Test that validate_data triggers load_dataset and inspects head() and describe()."""
    # Create mock DataFrame to verify .head() and .describe() calls
    mock_df = MagicMock()
    mock_load_dataset.return_value = mock_df

    validator_instance.validate_data()

    # Verify load_dataset was triggered
    mock_load_dataset.assert_called_once()
    assert validator_instance.data == mock_df

    # Verify inspection methods were called on dataset
    mock_df.head.assert_called_once()
    mock_df.describe.assert_called_once()


# ------------------------------------------------------------------------------
# 3. Class Attributes Verification
# ------------------------------------------------------------------------------

def test_default_file_path():
    """Verify default file path attribute."""
    assert ValidateData.file_path == "../../data/raw/raw_data.csv"