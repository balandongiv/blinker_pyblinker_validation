"""Session-level configuration objects for annotation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnnotationSession:
    """Configuration for a single annotation run."""

    fif_path: Path
    csv_path: Path
    start: float
    end: float
    annotated_before: bool
