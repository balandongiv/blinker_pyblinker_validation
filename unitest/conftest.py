"""Pytest configuration for ensuring local packages can be imported."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in {PROJECT_ROOT, PROJECT_ROOT / "src"}:
    sys.path.insert(0, str(path))
