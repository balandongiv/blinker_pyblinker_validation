"""Tests for the Google Drive downloader helper."""

from __future__ import annotations

from unittest import TestCase

from src.helper import gdrive_downloader


class DownloadDriveFolderIntegrationTests(TestCase):
    """Integration test that exercises the real Google Drive download."""

    def test_dataset_contains_expected_files(self) -> None:
        """The shared dataset is downloaded and contains key fixture files."""

        dataset_root = gdrive_downloader.get_dataset_root()

        try:
            files = gdrive_downloader.download_drive_folder(skip_existing=True)
        except RuntimeError as exc:
            message = str(exc)
            if "Failed to download folder" in message:
                self.skipTest(
                    "Google Drive download is not accessible in this environment: "
                    f"{message}"
                )
            raise

        self.assertTrue(files, "Download helper returned no files.")
        self.assertTrue(
            dataset_root.is_dir(),
            f"Dataset root directory is missing: {dataset_root}",
        )

        expected_paths = [
            dataset_root / "S1" / "S1.fif",
            dataset_root
            / "S1"
            / "S01_20170519_043933"
            / "seg_annotated_raw.fif",
            dataset_root
            / "S1"
            / "S01_20170519_043933"
            / "eeg_clean_epo.fif",
            dataset_root
            / "S1"
            / "S01_20170519_043933"
            / "seg_ear.fif",
            dataset_root
            / "S1"
            / "S01_20170519_043933"
            / "ear_eog.fif",
            dataset_root / "S1" / "S01_20170519_043933_2",
            dataset_root / "S1" / "S01_20170519_043933_3",
            dataset_root / "S2",
        ]

        for path in expected_paths:
            self.assertTrue(
                path.exists(),
                f"Expected dataset entry missing: {path}",
            )

