"""Download the murat_2018 dataset from the list of Figshare URLs.

The script reads ``murat_2018_dataset.txt`` (one URL per line), downloads the
referenced ``.mat`` files and stores them inside ``<root>/<recording_id>/``
folders.  A ``recording_id`` corresponds to the stem of the ``.mat`` filename.

Only the first three URLs are processed by default so that the development
workflow remains lightweight.  The ``--limit`` CLI flag (or the
``MURAT_DATASET_LIMIT`` environment variable) can be used to override the
behaviour and process the full list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT_RAW = os.environ.get("MURAT_DATASET_ROOT")
DEFAULT_ROOT = Path(DEFAULT_ROOT_RAW) if DEFAULT_ROOT_RAW else REPO_ROOT / "data" / "murat_2018"
DEFAULT_LIMIT_RAW = os.environ.get("MURAT_DATASET_LIMIT")
DEFAULT_LIMIT = int(DEFAULT_LIMIT_RAW) if DEFAULT_LIMIT_RAW is not None else 3
DEFAULT_DATASET_FILE = REPO_ROOT / "murat_2018_dataset.txt"

LOGGER = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when a download fails irrecoverably."""


@dataclass(slots=True)
class DownloadTask:
    """Container describing a single download request."""

    url: str
    destination: Path
    recording_id: str


def _iter_urls(file_path: Path) -> Iterable[str]:
    """Yield non-empty URLs from ``file_path``."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset list not found: {file_path}")

    with file_path.open("r", encoding="utf8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            yield cleaned


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_metadata(task: DownloadTask, size: int) -> None:
    metadata = {
        "url": task.url,
        "recording_id": task.recording_id,
        "path": str(task.destination),
        "bytes": size,
        "sha256": _compute_sha256(task.destination),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    target = task.destination.with_suffix(".metadata.json")
    with target.open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _download_file(url: str, destination: Path) -> int:
    LOGGER.info("Downloading %s → %s", url, destination)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    temp_path = destination.with_suffix(destination.suffix + ".part")
    total = 0
    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            total += len(chunk)

    temp_path.replace(destination)
    return total


def _validate_mat_file(path: Path) -> None:
    if not path.exists():
        raise DownloadError(f"Downloaded file missing: {path}")

    with path.open("rb") as handle:
        header = handle.read(128)

    if len(header) < 8:
        raise DownloadError(f"Downloaded file is empty or truncated: {path}")

    if not (header.startswith(b"MATLAB") or header[:4] == b"MATL"):
        raise DownloadError(f"Downloaded file is not a MATLAB MAT-file: {path}")


def _derive_filename(url: str) -> str:
    parsed = urlparse(url)
    candidate = Path(unquote(parsed.path)).name
    if not candidate:
        candidate = "recording"

    if not candidate.lower().endswith(".mat"):
        LOGGER.info("URL %s lacks a .mat suffix; saving as %s.mat", url, candidate)
        candidate = f"{candidate}.mat"

    return candidate


def _prepare_task(url: str, root: Path) -> DownloadTask | None:
    filename = _derive_filename(url)
    recording_id = Path(filename).stem
    folder = root / recording_id
    folder.mkdir(parents=True, exist_ok=True)

    destination = folder / filename
    return DownloadTask(url=url, destination=destination, recording_id=recording_id)


def _should_skip(task: DownloadTask) -> bool:
    if not task.destination.exists():
        return False
    size = task.destination.stat().st_size
    if size <= 0:
        LOGGER.warning("Existing file has zero bytes and will be re-downloaded: %s", task.destination)
        return False
    LOGGER.info("Skipping existing file (%s bytes): %s", size, task.destination)
    return True


def download_dataset(
    dataset_file: Path,
    root: Path = DEFAULT_ROOT,
    limit: int | None = DEFAULT_LIMIT,
    retries: int = 3,
) -> int:
    """Download ``.mat`` files described in ``dataset_file`` into ``root``.

    Returns the number of successful downloads (existing files count as
    successes).  A :class:`DownloadError` is raised when none of the requested
    files could be obtained.
    """

    root.mkdir(parents=True, exist_ok=True)

    success = 0
    total = 0
    for idx, url in enumerate(_iter_urls(dataset_file), start=1):
        if limit is not None and limit >= 0 and idx > limit:
            LOGGER.info("Limiter active (%s); stopping after %s URL(s)", limit, limit)
            break

        task = _prepare_task(url, root)
        if task is None:
            continue

        total += 1
        if _should_skip(task):
            success += 1
            continue

        attempt = 0
        while attempt < retries:
            attempt += 1
            try:
                size = _download_file(task.url, task.destination)
                _validate_mat_file(task.destination)
                _write_metadata(task, size)
            except Exception as exc:  # noqa: BLE001 - log and retry
                LOGGER.error(
                    "Failed to download %s on attempt %s/%s: %s",
                    task.url,
                    attempt,
                    retries,
                    exc,
                )
                if task.destination.exists():
                    task.destination.unlink(missing_ok=True)
                time.sleep(min(2**attempt, 10))
            else:
                LOGGER.info("Downloaded %s (%s bytes)", task.destination, size)
                success += 1
                break
        else:
            LOGGER.error("Giving up on %s after %s attempts", task.url, retries)

    if success == 0 and total == 0:
        raise DownloadError("No dataset URLs were processed from the dataset list")
    if success == 0:
        raise DownloadError("Failed to download any dataset files")

    return success


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="Text file that lists dataset URLs (default: repository root).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Destination root directory for the downloads.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of URLs to process (negative disables the limiter).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per URL before giving up.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    limit = None if args.limit is None or args.limit < 0 else args.limit

    try:
        count = download_dataset(
            dataset_file=args.dataset_file,
            root=args.root,
            limit=limit,
            retries=args.retries,
        )
    except Exception as exc:  # noqa: BLE001 - top-level exception handler
        LOGGER.error("Download failed: %s", exc)
        return 1

    LOGGER.info("Successfully processed %s file(s)", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
