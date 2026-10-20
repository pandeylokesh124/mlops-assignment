import os
import unittest
from unittest.mock import MagicMock, patch

# Import your class (adjust path depending on your directory structure)
# e.g., from src.feature.download_data import DownloadData
from src.data.download_data import DownloadData


class TestDownloadData(unittest.TestCase):

    def setUp(self):
        """Set up a fresh instance before each test."""
        self.downloader = DownloadData()

    @patch("src.data.download_data.fetch_ucirepo")
    @patch("src.data.download_data.urlretrieve")
    @patch("os.makedirs")
    def test_download_data_success(self, mock_makedirs, mock_urlretrieve, mock_fetch):
        """Test successful dataset fetching, directory creation, and file saving."""
        # Mock UCI repository object and metadata response
        mock_repo = MagicMock()
        mock_repo.metadata.data_url = "https://archive.ics.uci.edu/static/public/45/data.csv"
        mock_fetch.return_value = mock_repo

        # Execute method
        self.downloader.download_data()

        # Assertions
        mock_fetch.assert_called_once_with(id=45)
        mock_makedirs.assert_called_once_with(self.downloader.target_dir, exist_ok=True)

        expected_destination = os.path.join(self.downloader.target_dir, self.downloader.file_name)
        mock_urlretrieve.assert_called_once_with(
            "https://archive.ics.uci.edu/static/public/45/data.csv",
            expected_destination
        )

    @patch("src.data.download_data.fetch_ucirepo")
    @patch("builtins.print")
    def test_download_data_handles_fetch_exception(self, mock_print, mock_fetch):
        """Test that network/UCI repo exceptions are caught and printed gracefully."""
        # Simulate network or API error when calling fetch_ucirepo
        error_message = "Connection timeout reaching UCI repository"
        mock_fetch.side_effect = Exception(error_message)

        # Execute method (should not raise an unhandled exception)
        self.downloader.download_data()

        # Verify the exception message was printed
        mock_print.assert_called_with(mock_fetch.side_effect)

    @patch("src.data.download_data.fetch_ucirepo")
    @patch("src.data.download_data.urlretrieve")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_download_data_handles_download_exception(
            self, mock_print, mock_makedirs, mock_urlretrieve, mock_fetch
    ):
        """Test that file downloading exceptions are caught and printed gracefully."""
        mock_repo = MagicMock()
        mock_repo.metadata.data_url = "https://invalid-url.org/data.csv"
        mock_fetch.return_value = mock_repo

        # Simulate exception during urlretrieve
        download_error = Exception("HTTP 404: Not Found")
        mock_urlretrieve.side_effect = download_error

        # Execute method
        self.downloader.download_data()

        # Verify the exception was printed
        mock_print.assert_called_with(download_error)

    def test_default_attributes(self):
        """Test class default parameters."""
        self.assertEqual(DownloadData.target_dir, "../../data/raw")
        self.assertEqual(DownloadData.file_name, "raw_data.csv")


if __name__ == "__main__":
    unittest.main()