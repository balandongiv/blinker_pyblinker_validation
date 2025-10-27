"""Download and extract unit test data from Google Drive."""

from __future__ import annotations

from pathlib import Path
from typing import List
import shutil
import gdown

TEST_DATA_URL = "https://drive.google.com/drive/folders/1yysPglbZ_8c6HcI2CsWmW6vkh3vk7g0L?usp=sharing"


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

    downloaded = gdown.download_folder(url=url, output=str(destination), quiet=True)
    if downloaded is None:
        print("No files were downloaded.")
        return []

    paths: List[Path] = []
    for file_str in downloaded:
        path = Path(file_str)
        if path.suffix in {".zip", ".tar", ".gz"}:
            shutil.unpack_archive(str(path), str(destination))
            path.unlink()
            for inner in Path(destination).rglob("*"):
                if inner.is_file():
                    paths.append(inner.resolve())
        else:
            paths.append(path.resolve())

    print("Downloaded and extracted the following files:")
    for p in paths:
        print(f" - {p}")

    return paths


if __name__ == "__main__":
    download_test_files()
