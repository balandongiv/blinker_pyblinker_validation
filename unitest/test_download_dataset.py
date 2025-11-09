"""Tests for the murat dataset downloader."""
from __future__ import annotations


import json
from pathlib import Path

import pytest

from src.murat import download_dataset as downloader


class _DummyResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:  # noqa: D401 - part of requests API
        """Pretend the response succeeded."""

    def iter_content(self, chunk_size: int):  # noqa: D401 - part of requests API
        """Yield the fake payload in a single chunk."""

        yield self._payload


@pytest.mark.parametrize("url", [
    "https://figshare.com/ndownloader/files/12400412",
    "https://figshare.com/ndownloader/files/12400412?download=1",
])
def test_download_figshare_like_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str) -> None:
    """Ensure Figshare downloader URLs without ``.mat`` suffixes are supported."""

    payload = (
        b"MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Mon Jan 01 00:00:00 2018"
        + b" " * 64
    )

    def fake_get(request_url: str, *, stream: bool, timeout: int):
        assert request_url == url
        assert stream is True
        assert timeout == 30
        return _DummyResponse(payload)

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    dataset_file = tmp_path / "urls.txt"
    dataset_file.write_text(f"{url}\n", encoding="utf8")

    root = tmp_path / "dataset"
    count = downloader.download_dataset(dataset_file=dataset_file, root=root, limit=None)

    assert count == 1

    subfolders = list(root.iterdir())
    assert len(subfolders) == 1
    recording_dir = subfolders[0]
    mat_files = list(recording_dir.glob("*.mat"))
    assert len(mat_files) == 1
    assert mat_files[0].read_bytes().startswith(b"MATLAB")

    metadata_files = list(recording_dir.glob("*.metadata.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf8"))
    assert metadata["bytes"] == len(payload)
    assert metadata["url"] == url
