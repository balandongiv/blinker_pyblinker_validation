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
            target_dir = Path(output)
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / "example.txt"
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
            existing_file = output_path / "existing.txt"
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
