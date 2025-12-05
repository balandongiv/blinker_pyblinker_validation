"""Shared constants for the annotation UI package."""

from __future__ import annotations

from pathlib import Path

DATASET_PATHS_FILE = Path(__file__).resolve().parents[2] / "config" / "dataset_paths.json"

ANNOTATION_COLUMNS = ["onset", "duration", "description"]
DEFAULT_ROOT_CANDIDATES = [
    Path(r"D:\dataset\murat_2018"),
    Path(r"G:\Other computers\My Computer\murat_2018"),
]
DEFAULT_ROOT = DEFAULT_ROOT_CANDIDATES[0]