"""Integration test for the Google Drive downloader helper."""

from __future__ import annotations

from unittest import TestCase

from src.helper import gdrive_downloader


class TestDatasetDownload(TestCase):
    """Ensure the real Google Drive dataset is downloaded and complete."""

    def test_dataset_download_contains_required_entries(self) -> None:
        """The dataset download yields all documented files and directories."""

        dataset_root = gdrive_downloader.get_dataset_root()

        files = gdrive_downloader.download_drive_folder()

        self.assertTrue(files, "Download helper returned no files.")
        self.assertTrue(
            dataset_root.is_dir(),
            f"Dataset root directory is missing: {dataset_root}",
        )

        for relative_path in gdrive_downloader.REQUIRED_DATASET_ENTRIES:
            path = dataset_root / relative_path
            with self.subTest(path=path):
                self.assertTrue(
                    path.exists(),
                    f"Expected dataset entry missing: {path}",
                )
