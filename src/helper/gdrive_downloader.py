"""Utilities to download Google Drive folders used in tests.

The module exposes :func:`download_drive_folder` which wraps :mod:`gdown`
functionality and adds a couple of niceties:

* sensible defaults for this repository,
* the ability to skip a download when files already exist, and
* basic error handling so tests can provide actionable failures.

The helper is primarily used by the unit tests to obtain large fixtures that
should not live in the Git history.  The default download directory is ignored
by Git (see ``.gitignore``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import gdown

DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/10lRz7p6YxftylNlrZ5GW645fwrKBzhLM?usp=sharing"
DEFAULT_DOWNLOAD_DIR = Path(__file__).resolve().parents[2] / "download_cache" / "gdrive_folder"


def _normalise_paths(paths: Iterable[str | Path], root: Path) -> List[Path]:
    """Return paths as :class:`Path` objects rooted at *root* when needed."""

    normalised: List[Path] = []
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        normalised.append(path)
    return normalised


def download_drive_folder(
    folder_url: str = DEFAULT_FOLDER_URL,
    output_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    *,
    skip_existing: bool = True,
    verify_download: bool = True,
) -> List[Path]:
    """Download the Google Drive *folder_url* to *output_dir*.

    Parameters
    ----------
    folder_url:
        Public Google Drive folder URL to download from.
    output_dir:
        Directory where the contents should be extracted.
    skip_existing:
        When :data:`True`, the download is skipped if *output_dir* already
        contains files.
    verify_download:
        When :data:`True`, the function verifies that the returned paths exist.

    Returns
    -------
    list[pathlib.Path]
        Paths to the downloaded files.

    Raises
    ------
    RuntimeError
        If ``gdown`` fails to fetch the folder or the verification fails.
    """

    output_path = Path(output_dir)

    if skip_existing and output_path.exists():
        existing_files = sorted(p for p in output_path.rglob("*") if p.is_file())
        if existing_files:
            return existing_files

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = gdown.download_folder(
            folder_url,
            quiet=True,
            use_cookies=False,
            output=str(output_path),
        )
    except Exception as exc:  # pragma: no cover - pass through the original error
        raise RuntimeError(f"Failed to download folder from {folder_url!r}") from exc

    downloaded = downloaded or []
    normalised = _normalise_paths(downloaded, output_path)

    if verify_download:
        missing = [path for path in normalised if not path.exists()]
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            raise RuntimeError(f"Download incomplete, missing files: {missing_str}")

    return normalised


def main() -> None:
    """CLI entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="Download Google Drive test data")
    parser.add_argument(
        "--url",
        default=DEFAULT_FOLDER_URL,
        help="Google Drive folder URL (defaults to the repository test data folder)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Where to download the folder (defaults to download_cache/gdrive_folder)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Force a re-download even if files already exist",
    )
    args = parser.parse_args()

    files = download_drive_folder(
        folder_url=args.url,
        output_dir=args.output,
        skip_existing=not args.no_skip,
    )

    print(f"Downloaded {len(files)} files to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()
