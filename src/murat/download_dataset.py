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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import requests

from src.utils.config_utils import (
    DEFAULT_CONFIG_PATH,
    get_path_setting,
    load_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_ROOT = get_path_setting(CONFIG, "download_root", env_var="MURAT_DATASET_ROOT")
DEFAULT_LIMIT_RAW = os.environ.get("MURAT_DATASET_LIMIT")
DEFAULT_LIMIT = int(DEFAULT_LIMIT_RAW) if DEFAULT_LIMIT_RAW is not None else 3
DEFAULT_DATASET_FILE = get_path_setting(CONFIG, "dataset_file")

LOGGER = logging.getLogger(__name__)
CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = (10, 120)
MATLAB_73_SIGNATURE = b"\x89HDF\r\n\x1a\n"
TEXTUAL_CONTENT_TYPES = {
    "application/json",
    "application/problem+json",
    "application/xml",
    "text/html",
    "text/plain",
    "text/xml",
}
REQUEST_HEADERS = {
    "Accept": "*/*",
    "User-Agent": "blinker-pyblinker-validation/figshare-downloader",
}
FIGSHARE_PUBLIC_DOWNLOAD_HOST = "ndownloader.figshare.com"


class DownloadError(RuntimeError):
    """Raised when a download fails irrecoverably."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(slots=True)
class DownloadTask:
    """Container describing a single download request."""

    url: str
    destination: Path
    recording_id: str


@dataclass(slots=True)
class DownloadResult:
    """Details captured from a completed HTTP download."""

    bytes_written: int
    destination: Path
    status_code: int
    request_url: str
    final_url: str
    content_type: str | None
    content_length: int | None
    response_headers: dict[str, str]


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


def _write_metadata(task: DownloadTask, result: DownloadResult) -> None:
    metadata = {
        "content_length": result.content_length,
        "content_type": result.content_type,
        "final_url": result.final_url,
        "http_status": result.status_code,
        "recording_id": task.recording_id,
        "path": str(result.destination),
        "bytes": result.bytes_written,
        "request_url": result.request_url,
        "response_headers": result.response_headers,
        "sha256": _compute_sha256(result.destination),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": task.url,
    }
    target = result.destination.with_suffix(".metadata.json")
    with target.open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _normalize_content_type(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(";", 1)[0].strip().lower() or None


def _parse_content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        LOGGER.warning("Ignoring non-numeric Content-Length value: %r", value)
        return None
    if parsed < 0:
        LOGGER.warning("Ignoring negative Content-Length value: %r", value)
        return None
    return parsed


def _sanitize_preview(chunk: bytes) -> str:
    preview = chunk[:160].decode("utf8", errors="replace")
    return re.sub(r"\s+", " ", preview).strip()


def _ensure_mat_suffix(filename: str) -> str:
    candidate = Path(filename).name.strip()
    if not candidate:
        candidate = "recording"
    if not candidate.lower().endswith(".mat"):
        candidate = f"{candidate}.mat"
    return candidate


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None

    extended = re.search(r"filename\*\s*=\s*([^']*)''([^;]+)", value, flags=re.IGNORECASE)
    if extended:
        encoded_name = extended.group(2).strip().strip('"')
        decoded = unquote(encoded_name)
        return Path(decoded).name or None

    basic = re.search(r'filename\s*=\s*"([^"]+)"', value, flags=re.IGNORECASE)
    if basic:
        return Path(basic.group(1).strip()).name or None

    basic_unquoted = re.search(r"filename\s*=\s*([^;]+)", value, flags=re.IGNORECASE)
    if basic_unquoted:
        return Path(basic_unquoted.group(1).strip().strip('"')).name or None

    return None


def _looks_like_text_response(preview: bytes) -> bool:
    stripped = preview.lstrip().lower()
    if not stripped:
        return False
    return stripped.startswith(
        (
            b"<!doctype html",
            b"<html",
            b"<?xml",
            b"{",
            b"[",
        )
    )


def _validate_response_payload(
    *,
    final_url: str,
    content_type: str | None,
    preview: bytes,
) -> None:
    if preview.startswith(b"MATLAB") or preview.startswith(MATLAB_73_SIGNATURE):
        return

    if content_type in TEXTUAL_CONTENT_TYPES or _looks_like_text_response(preview):
        preview_text = _sanitize_preview(preview)
        raise DownloadError(
            "Server returned a non-file response "
            f"(content-type={content_type or 'missing'}) from {final_url}: {preview_text!r}",
            retryable=False,
        )


def _normalize_figshare_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return url

    path_parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.lower() == "figshare.com" and path_parts[:2] == ["ndownloader", "files"] and len(path_parts) >= 3:
        file_id = path_parts[2]
        normalized = f"https://{FIGSHARE_PUBLIC_DOWNLOAD_HOST}/files/{file_id}"
        if url != normalized:
            LOGGER.info("Normalizing Figshare download URL from %s to %s", url, normalized)
        return normalized

    return url


def _derive_filename_from_response(
    *,
    original_url: str,
    final_url: str,
    headers: dict[str, str],
) -> str:
    content_disposition = headers.get("content-disposition")
    filename = _filename_from_content_disposition(content_disposition)
    if filename:
        return _ensure_mat_suffix(filename)

    final_candidate = Path(unquote(urlparse(final_url).path)).name
    if final_candidate:
        return _ensure_mat_suffix(final_candidate)

    return _derive_filename(original_url)


def _find_existing_download(task: DownloadTask) -> Path | None:
    if task.destination.exists():
        return task.destination

    if not task.destination.parent.exists():
        return None

    for candidate in sorted(task.destination.parent.glob("*.mat")):
        try:
            _validate_mat_file(candidate)
        except DownloadError:
            continue
        return candidate

    return None


def _looks_like_figshare_browser_challenge(
    *,
    url: str,
    status_code: int,
    headers: dict[str, str],
    content_type: str | None,
    content_length: int | None,
) -> bool:
    parsed = urlparse(url)
    waf_action = headers.get("x-amzn-waf-action", "").lower()
    is_figshare = "figshare.com" in parsed.netloc.lower()
    return (
        is_figshare
        and status_code == 202
        and waf_action == "challenge"
        and content_type == "text/html"
        and (content_length in {None, 0})
    )


def _should_ignore_env_proxy() -> bool:
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        value = os.environ.get(key)
        if not value:
            continue
        parsed = urlparse(value if "://" in value else f"http://{value}")
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9:
            return True
    return False


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    if _should_ignore_env_proxy():
        session.trust_env = False
        LOGGER.warning(
            "Ignoring HTTP(S) proxy environment variables because they point to a disabled loopback proxy"
        )
    return session


def _download_file(session: requests.Session, url: str, destination: Path) -> DownloadResult:
    request_url = _normalize_figshare_url(url)
    LOGGER.info("Downloading %s → %s", request_url, destination)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    temp_path.unlink(missing_ok=True)

    try:
        with session.get(request_url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True) as response:
            response.raise_for_status()
            content_type = _normalize_content_type(response.headers.get("Content-Type"))
            content_length = _parse_content_length(response.headers.get("Content-Length"))
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            LOGGER.info(
                "HTTP %s final_url=%s content_type=%s content_length=%s waf_action=%s",
                response.status_code,
                response.url,
                content_type or "<missing>",
                content_length if content_length is not None else "<missing>",
                response_headers.get("x-amzn-waf-action", "<missing>"),
            )
            if response.history:
                chain = " -> ".join(
                    f"{item.status_code}:{item.headers.get('Location', item.url)}" for item in response.history
                )
                LOGGER.debug("Redirect chain for %s: %s", url, chain)

            if _looks_like_figshare_browser_challenge(
                url=response.url,
                status_code=response.status_code,
                headers=response_headers,
                content_type=content_type,
                content_length=content_length,
            ):
                raise DownloadError(
                    "Figshare returned an AWS WAF browser challenge instead of a file "
                    f"(status={response.status_code}, final_url={response.url})",
                    retryable=False,
                )

            resolved_filename = _derive_filename_from_response(
                original_url=url,
                final_url=response.url,
                headers=response_headers,
            )
            final_destination = destination.with_name(resolved_filename)
            if final_destination != destination:
                LOGGER.info("Resolved download filename to %s", final_destination.name)

            total = 0
            preview = b""
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    if not preview:
                        preview = chunk[:512]
                        _validate_response_payload(
                            final_url=response.url,
                            content_type=content_type,
                            preview=preview,
                        )
                    handle.write(chunk)
                    total += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if total <= 0:
        temp_path.unlink(missing_ok=True)
        raise DownloadError(f"Downloaded zero bytes from {url}")

    if content_length is not None and total != content_length:
        temp_path.unlink(missing_ok=True)
        raise DownloadError(
            f"Downloaded {total} bytes but expected {content_length} bytes from {response.url}"
        )

    final_destination.unlink(missing_ok=True)
    temp_path.replace(final_destination)
    return DownloadResult(
        bytes_written=total,
        destination=final_destination,
        status_code=response.status_code,
        request_url=request_url,
        final_url=response.url,
        content_type=content_type,
        content_length=content_length,
        response_headers=response_headers,
    )


def _validate_mat_file(path: Path) -> None:
    if not path.exists():
        raise DownloadError(f"Downloaded file missing: {path}")

    with path.open("rb") as handle:
        header = handle.read(128)

    if len(header) < 8:
        raise DownloadError(f"Downloaded file is empty or truncated ({len(header)} bytes): {path}")

    if not (header.startswith(b"MATLAB") or header[:4] == b"MATL" or header.startswith(MATLAB_73_SIGNATURE)):
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
    existing = _find_existing_download(task)
    if existing is None:
        return False
    if existing != task.destination:
        task.destination = existing
        task.recording_id = existing.stem
    size = existing.stat().st_size
    if size <= 0:
        LOGGER.warning("Existing file has zero bytes and will be re-downloaded: %s", existing)
        return False
    try:
        _validate_mat_file(existing)
    except DownloadError as exc:
        LOGGER.warning("Existing file will be re-downloaded because it is invalid: %s", exc)
        return False
    LOGGER.info("Skipping existing valid file (%s bytes): %s", size, existing)
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
    with _create_session() as session:
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
                    result = _download_file(session, task.url, task.destination)
                    task.destination = result.destination
                    task.recording_id = result.destination.stem
                    _validate_mat_file(result.destination)
                    _write_metadata(task, result)
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
                    if isinstance(exc, DownloadError) and not exc.retryable:
                        LOGGER.error("Not retrying %s because the failure is not retryable", task.url)
                        break
                    time.sleep(min(2**attempt, 10))
                else:
                    LOGGER.info(
                        "Downloaded %s (%s bytes, content_type=%s)",
                        result.destination,
                        result.bytes_written,
                        result.content_type or "<missing>",
                    )
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
