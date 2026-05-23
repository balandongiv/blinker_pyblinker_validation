"""Compatibility wrapper for Raja tutorial helpers.

These helpers now live under :mod:`src.ui_raja.cvat_helpers` so the reusable
logic remains in ``src`` while older tutorial scripts can keep importing the
same symbols from ``tutorial.raja_sequence.helper``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# from src.ui_raja.cvat_helpers import (  # noqa: E402,F401
#     filter_min_labels,
#     load_actual_annotations,
#     load_ground_truth,
#     match_ground_truth_to_annotations,
#     restructure_blink_dataframe,
#     unzip_file,
# )
