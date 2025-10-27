"""Integration tests for downloading Google Drive fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from src.helper import download_test_files

REQUIRED_DATASET_ENTRIES = [
    Path("eog_eeg_data") / "S1" / "S1.fif",
    Path("eog_eeg_data") / "S1" / "S01_20170519_043933" / "seg_annotated_raw.fif",
    Path("eog_eeg_data") / "S1" / "S01_20170519_043933" / "eeg_clean_epo.fif",
    Path("eog_eeg_data") / "S1" / "S01_20170519_043933" / "seg_ear.fif",
    Path("eog_eeg_data") / "S1" / "S01_20170519_043933" / "ear_eog.fif",
    Path("eog_eeg_data") / "S1" / "S01_20170519_043933_2",
    Path("eog_eeg_data") / "S1" / "S01_20170519_043933_3",
    Path("eog_eeg_data") / "S2",
]


class TestDatasetDownload(TestCase):
    """Ensure the Google Drive dataset can be retrieved and verified."""

    def setUp(self) -> None:
        self.destination = Path("unitest/test_files")

    def test_dataset_download_contains_required_entries(self) -> None:
        """Downloading the dataset yields all required files and folders."""

        files = download_test_files(destination=self.destination)
        self.assertTrue(files, "Download helper returned no files.")

        for relative_path in REQUIRED_DATASET_ENTRIES:
            path = self.destination / relative_path
            with self.subTest(path=path):
                self.assertTrue(
                    path.exists(),
                    f"Expected dataset entry missing: {path}",
                )

    def test_existing_dataset_is_verified(self) -> None:
        """Re-downloading should still ensure the expected entries exist."""

        download_test_files(destination=self.destination)
        files = download_test_files(destination=self.destination)
        self.assertTrue(files, "Download helper returned no files on re-run.")

        for relative_path in REQUIRED_DATASET_ENTRIES:
            path = self.destination / relative_path
            with self.subTest(path=path):
                self.assertTrue(
                    path.exists(),
                    f"Expected dataset entry missing after re-download: {path}",
                )
