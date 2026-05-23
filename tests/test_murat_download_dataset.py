from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from src.utils.download_dataset import (
    MATLAB_73_SIGNATURE,
    DownloadError,
    _download_file,
    _filename_from_content_disposition,
    _normalize_figshare_url,
    _should_ignore_env_proxy,
    _should_skip,
    _validate_mat_file,
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        url: str,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        history: list[object] | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.headers = headers or {}
        self._chunks = chunks or []
        self.history = history or []

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 0):
        del chunk_size
        yield from self._chunks


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.last_url: str | None = None
        self.last_kwargs: dict[str, object] | None = None

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.last_url = url
        self.last_kwargs = kwargs
        return self._response


@contextmanager
def workspace_tempdir():
    root = Path.cwd() / "scratch_test_cases" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class DownloadDatasetTests(unittest.TestCase):
    def test_filename_from_content_disposition_prefers_figshare_name(self) -> None:
        self.assertEqual(
            _filename_from_content_disposition(
                'attachment; filename="CLA-SubjectJ-170508-3St-LRHand-Inter.mat"'
            ),
            "CLA-SubjectJ-170508-3St-LRHand-Inter.mat",
        )

    def test_normalize_figshare_url_uses_public_download_host(self) -> None:
        self.assertEqual(
            _normalize_figshare_url("https://figshare.com/ndownloader/files/12400412"),
            "https://ndownloader.figshare.com/files/12400412",
        )

    def test_download_file_rejects_html_body(self) -> None:
        with workspace_tempdir() as tmpdir:
            destination = tmpdir / "sample.mat"
            response = FakeResponse(
                url="https://files.figshare.com/sample",
                headers={
                    "Content-Length": "20",
                    "Content-Type": "text/html; charset=utf-8",
                },
                chunks=[b"<html>blocked</html>"],
            )

            with self.assertRaisesRegex(DownloadError, "non-file response") as exc_info:
                _download_file(FakeSession(response), "https://figshare.com/ndownloader/files/1", destination)

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".mat.part").exists())
            self.assertFalse(exc_info.exception.retryable)

    def test_download_file_detects_content_length_mismatch(self) -> None:
        with workspace_tempdir() as tmpdir:
            destination = tmpdir / "sample.mat"
            body = b"MATLAB 5.0 MAT-file"
            response = FakeResponse(
                url="https://files.figshare.com/sample",
                headers={
                    "Content-Length": str(len(body) + 10),
                    "Content-Type": "application/octet-stream",
                },
                chunks=[body],
            )

            with self.assertRaisesRegex(DownloadError, "expected"):
                _download_file(FakeSession(response), "https://figshare.com/ndownloader/files/1", destination)

            self.assertFalse(destination.exists())

    def test_download_file_uses_figshare_filename_from_content_disposition(self) -> None:
        with workspace_tempdir() as tmpdir:
            destination = tmpdir / "12400412.mat"
            body = b"MATLAB 5.0 MAT-file" + (b"\x00" * 200)
            response = FakeResponse(
                url="https://files.figshare.com/final/12400412",
                headers={
                    "Content-Length": str(len(body)),
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": 'attachment; filename="CLA-SubjectJ-170508-3St-LRHand-Inter.mat"',
                },
                chunks=[body],
                history=[FakeResponse(status_code=302, url="https://figshare.com/ndownloader/files/12400412")],
            )
            session = FakeSession(response)

            result = _download_file(
                session,
                "https://figshare.com/ndownloader/files/12400412",
                destination,
            )

            self.assertEqual(result.bytes_written, len(body))
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.final_url, "https://files.figshare.com/final/12400412")
            self.assertEqual(result.content_type, "application/octet-stream")
            self.assertEqual(result.content_length, len(body))
            self.assertEqual(
                result.destination.name,
                "CLA-SubjectJ-170508-3St-LRHand-Inter.mat",
            )
            self.assertTrue(result.destination.exists())
            self.assertFalse(destination.exists())
            self.assertEqual(session.last_url, "https://ndownloader.figshare.com/files/12400412")

    def test_download_file_rejects_figshare_waf_challenge(self) -> None:
        with workspace_tempdir() as tmpdir:
            destination = tmpdir / "sample.mat"
            response = FakeResponse(
                status_code=202,
                url="https://figshare.com/ndownloader/files/12400412",
                headers={
                    "Content-Length": "0",
                    "Content-Type": "text/html; charset=utf-8",
                    "x-amzn-waf-action": "challenge",
                },
                chunks=[],
            )

            with self.assertRaisesRegex(DownloadError, "AWS WAF browser challenge") as exc_info:
                _download_file(FakeSession(response), "https://figshare.com/ndownloader/files/12400412", destination)

            self.assertFalse(exc_info.exception.retryable)

    def test_validate_mat_file_accepts_hdf5_mat_signature(self) -> None:
        with workspace_tempdir() as tmpdir:
            target = tmpdir / "sample.mat"
            target.write_bytes(MATLAB_73_SIGNATURE + (b"\x00" * 128))

            _validate_mat_file(target)

    def test_should_skip_rejects_invalid_existing_file(self) -> None:
        with workspace_tempdir() as root:
            folder = root / "sample"
            folder.mkdir()
            target = folder / "sample.mat"
            target.write_text("<html>error</html>", encoding="utf8")

            from src.utils.download_dataset import DownloadTask

            task = DownloadTask(
                url="https://figshare.com/ndownloader/files/1",
                destination=target,
                recording_id="sample",
            )

            self.assertFalse(_should_skip(task))

    def test_should_ignore_env_proxy_only_for_disabled_loopback_proxy(self) -> None:
        with patch.dict("os.environ", {"HTTPS_PROXY": "http://127.0.0.1:9"}, clear=True):
            self.assertTrue(_should_ignore_env_proxy())

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy.example.com:8080"}, clear=True):
            self.assertFalse(_should_ignore_env_proxy())


if __name__ == "__main__":
    unittest.main()
