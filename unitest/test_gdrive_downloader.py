"""Tests for the Google Drive downloader helper."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase
from unittest.mock import patch

from src.helper import gdrive_downloader


class DownloadDriveFolderTests(TestCase):
    """Test the :func:`download_drive_folder` helper."""

    def test_successful_download_creates_files(self) -> None:
        """A successful download returns the created files."""

        def fake_download(url: str, quiet: bool, use_cookies: bool, output: str) -> List[str]:
            target_root = Path(output)
            dataset_root = target_root / gdrive_downloader.DATASET_FOLDER_NAME
            dataset_root.mkdir(parents=True, exist_ok=True)
            file_path = dataset_root / "example.txt"
            file_path.write_text("payload", encoding="utf-8")
            return [str(file_path)]

        with TemporaryDirectory() as tmpdir:
            with patch(
                "src.helper.gdrive_downloader.gdown.download_folder",
                side_effect=fake_download,
            ) as mock_download:
                downloaded = gdrive_downloader.download_drive_folder(
                    folder_url="https://example.com/folder",
                    output_dir=tmpdir,
                    skip_existing=False,
                )

            mock_download.assert_called_once()
            self.assertEqual(1, len(downloaded))
            self.assertTrue(downloaded[0].exists())
            self.assertEqual("payload", downloaded[0].read_text(encoding="utf-8"))

    def test_skip_existing_returns_existing_files(self) -> None:
        """When files exist and skipping is enabled, no download is attempted."""

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            dataset_root = output_path / gdrive_downloader.DATASET_FOLDER_NAME
            dataset_root.mkdir(parents=True, exist_ok=True)
            existing_file = dataset_root / "existing.txt"
            existing_file.write_text("cached", encoding="utf-8")

            with patch("src.helper.gdrive_downloader.gdown.download_folder") as mock_download:
                downloaded = gdrive_downloader.download_drive_folder(
                    folder_url="https://example.com/folder",
                    output_dir=tmpdir,
                    skip_existing=True,
                )

        mock_download.assert_not_called()
        self.assertEqual([existing_file], downloaded)

    def test_failed_download_raises_runtime_error(self) -> None:
        """Any failure from ``gdown`` is surfaced as :class:`RuntimeError`."""

        with TemporaryDirectory() as tmpdir, patch(
            "src.helper.gdrive_downloader.gdown.download_folder",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                gdrive_downloader.download_drive_folder(
                    folder_url="https://example.com/folder",
                    output_dir=tmpdir,
                    skip_existing=False,
                )


class DownloadDriveFolderIntegrationTests(TestCase):
    """Integration tests that exercise the real Google Drive download."""

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

        expected_files = [
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
        ]

        for path in expected_files:
            self.assertTrue(
                path.exists(),
                f"Expected dataset file missing: {path}",
            )

        expected_directories = [
            dataset_root / "S1",
            dataset_root / "S1" / "S01_20170519_043933",
            dataset_root / "S1" / "S01_20170519_043933_2",
            dataset_root / "S1" / "S01_20170519_043933_3",
            dataset_root / "S2",
        ]

        for directory in expected_directories:
            self.assertTrue(
                directory.is_dir(),
                f"Expected dataset directory missing: {directory}",
            )

