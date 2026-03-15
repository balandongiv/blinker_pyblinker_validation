"""Shared path helpers for validation scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
VALIDATION_ROOT = SRC_ROOT / "validation"
REPORTS_ROOT = REPO_ROOT / "reports"
REPORTS_DIR = REPORTS_ROOT / "validation"
FINDINGS_DIR = REPO_ROOT / "docs" / "findings"
SUMMARY_METRICS_PATH = VALIDATION_ROOT / "summary_metrics.csv"

