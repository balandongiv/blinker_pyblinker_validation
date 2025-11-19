"""Discovery helpers for locating FIF files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def find_fif_files(root: Path, provided: Iterable[str] | None = None) -> list[Path]:
    """Return available FIF files from an explicit list or a directory search."""

    if provided:
        paths = [Path(p) for p in provided]
    else:
        if not root.exists():
            raise FileNotFoundError(
                f"The root path {root} does not exist; please verify the dataset location."
            )
        candidates = list(root.rglob("*.fif")) + list(root.rglob("*.fif.gz"))
        seen: set[Path] = set()
        paths: list[Path] = []
        for path in sorted(candidates):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No FIF files were found in the specified inputs.")
    return existing
