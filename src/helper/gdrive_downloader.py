"""Download and extract unit test data from Google Drive."""

from __future__ import annotations

from pathlib import Path
from typing import List
import shutil
import gdown

TEST_DATA_URL = "https://drive.google.com/file/d/1z9zoMWTHXD3L5lYHu-Vgh7aaH74Kdnd9/view?usp=sharing"


def download_test_files(destination: Path = Path("unitest/test_files"), url: str = TEST_DATA_URL) -> List[Path]:
    """Download test data from Google Drive and print full paths.

    Parameters
    ----------
    destination : Path, optional
        Folder where the downloaded files will be stored.
    url : str, optional
        Public Google Drive folder URL to download.

    Returns
    -------
    List[Path]
        Absolute paths to the downloaded and extracted files.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Downloading test files to: {destination}")

    downloaded = gdown.download(url=url, output=str(destination), quiet=True, fuzzy=True)

    downloaded_paths: List[Path]
    if downloaded is None:
        print("Download skipped or no new files were retrieved. Verifying existing files.")
        downloaded_paths = []
    else:
        if isinstance(downloaded, list):
            downloaded_paths = [Path(item) for item in downloaded]
        else:
            downloaded_paths = [Path(downloaded)]

    for path in downloaded_paths:
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in {".zip", ".tar", ".gz", ".bz2"}:
            try:
                shutil.unpack_archive(str(path), str(destination))
            finally:
                if path.exists():
                    path.unlink()

    paths = sorted(inner.resolve() for inner in destination.rglob("*") if inner.is_file())

    if paths:
        print("Downloaded and discovered the following files:")
        for p in paths:
            print(f" - {p}")
    else:
        print("No files were found in the destination directory.")

    return paths


if __name__ == "__main__":
    download_test_files()
