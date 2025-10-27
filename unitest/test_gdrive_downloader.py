"""Tests for the Google Drive downloader helper."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from src.helper import gdrive_downloader


class DownloadDriveFolderTests(TestCase):
    """Tests that focus on successful downloads."""

    def test_download_success_returns_files(self) -> None:
        """A successful download yields file paths that exist on disk."""

        def fake_download(url: str, quiet: bool, use_cookies: bool, output: str):
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

    def test_dataset_downloads_and_contains_required_entries(self) -> None:
        """The real dataset can be downloaded and key files are present."""

        dataset_root = gdrive_downloader.get_dataset_root()

        files = gdrive_downloader.download_drive_folder(skip_existing=False)

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

