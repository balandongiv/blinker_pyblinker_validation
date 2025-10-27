"""Integration test for the Google Drive downloader helper."""

from __future__ import annotations

from unittest import TestCase

from src.helper import gdrive_downloader


class TestDatasetDownload(TestCase):
    """Ensure the real Google Drive dataset is downloaded and complete."""

    def test_dataset_download_contains_required_entries(self) -> None:
        """The dataset download yields all documented files and directories."""

        dataset_root = gdrive_downloader.get_dataset_root()

        files = gdrive_downloader.download_drive_folder(skip_existing=False)

        self.assertTrue(files, "Download helper returned no files.")
        self.assertTrue(
            dataset_root.is_dir(),
            f"Dataset root directory is missing: {dataset_root}",
        )

        expected_paths = [
            dataset_root / "S1" / "S1.fif",
            dataset_root / "S1" / "S01_20170519_043933" / "seg_annotated_raw.fif",
            dataset_root / "S1" / "S01_20170519_043933" / "eeg_clean_epo.fif",
            dataset_root / "S1" / "S01_20170519_043933" / "seg_ear.fif",
            dataset_root / "S1" / "S01_20170519_043933" / "ear_eog.fif",
            dataset_root / "S1" / "S01_20170519_043933_2",
            dataset_root / "S1" / "S01_20170519_043933_3",
            dataset_root / "S2",
        ]

        for path in expected_paths:
            with self.subTest(path=path):
                self.assertTrue(
                    path.exists(),
                    f"Expected dataset entry missing: {path}",
                )
