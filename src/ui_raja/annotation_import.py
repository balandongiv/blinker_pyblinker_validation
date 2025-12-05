"""Annotation import helpers for Raja sessions."""

from __future__ import annotations

import json
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from raja_sequence.helper import load_ground_truth, restructure_blink_dataframe, unzip_file

from .constants import ANNOTATION_COLUMNS, DEFAULT_SAMPLING_RATE
from .discovery import SessionInfo

logger = logging.getLogger(__name__)


class AnnotationImportError(Exception):
    """Raised when annotations cannot be imported for a session."""


def expected_zip_path(cvat_root: Path, session: SessionInfo) -> Path:
    return cvat_root / session.subject_id / "from_cvat" / f"{session.session_name}.zip"


def _discover_cvat_root(session: SessionInfo, provided_root: Path) -> Path:
    """Return the best CVAT root for a session, preferring the provided root."""

    provided_root = provided_root.resolve()
    if provided_root.exists():
        return provided_root

    resolved_fif = session.fif_path.resolve()
    for parent in resolved_fif.parents:
        candidate = parent / "CVAT_visual_annotation" / "cvat_zip_final"
        if candidate.exists():
            logger.info(
                "Using fallback CVAT root %s for %s/%s", candidate, session.subject_id, session.session_name
            )
            return candidate

    return provided_root


def load_shift_value(config_path: Path, session: SessionInfo) -> int:
    if not config_path.exists():
        logger.warning("Config file %s not found; using zero shift.", config_path)
        return 0
    try:
        content = json.loads(config_path.read_text())
        return int(content["data"][session.subject_id]["shift_cvat"][session.session_name])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read shift from %s: %s", config_path, exc)
        return 0


def build_annotation_frame(csv_path: Path, shift: int, sampling_rate: float) -> pd.DataFrame:
    df_gt = load_ground_truth(str(csv_path), shift, sampling_rate)
    df_gt = restructure_blink_dataframe(df_gt, sampling_rate)
    if df_gt.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    df_gt["start_sec"] = (df_gt["start"] / sampling_rate).abs()
    df_gt["duration_seconds"] = df_gt["duration_seconds"].abs()

    return pd.DataFrame(
        {
            "onset": df_gt["start_sec"].to_numpy(float),
            "duration": df_gt["duration_seconds"].to_numpy(float),
            "description": df_gt["blink_type"].astype(str).tolist(),
        }
    )


def _backup_annotation_source(source: Path, session: SessionInfo) -> Path:
    """Persist a backup of the original annotation alongside the FIF file."""

    backup_dir = session.backups_dir
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{session.fif_path.stem}_{timestamp}{source.suffix}"
    destination = backup_dir / backup_name
    shutil.copy2(source, destination)
    logger.info("Created annotation backup at %s", destination)
    return destination


def import_annotations(
    session: SessionInfo, cvat_root: Path, *, sampling_rate: float = DEFAULT_SAMPLING_RATE
) -> tuple[pd.DataFrame, Path]:
    """Import annotations from the CVAT ZIP and return a normalized frame and source path."""

    resolved_root = _discover_cvat_root(session, cvat_root)
    zip_path = expected_zip_path(resolved_root, session)
    if not zip_path.exists():
        logger.warning(
            "No CVAT ZIP available for %s/%s at %s", session.subject_id, session.session_name, zip_path
        )
        raise AnnotationImportError(f"Missing CVAT ZIP for {session.subject_id}/{session.session_name}: {zip_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        unzip_file(str(zip_path), tmpdir)
        csv_candidate: Optional[Path] = None
        for candidate in Path(tmpdir).rglob("default-annotations-human-imagelabels.csv"):
            csv_candidate = candidate
            break
        if csv_candidate is None:
            raise AnnotationImportError("No default-annotations-human-imagelabels.csv found in ZIP")

        shift_value = load_shift_value(Path("config_video_detail.json"), session)
        frame = build_annotation_frame(csv_candidate, shift_value, sampling_rate)
        return frame, zip_path


def ensure_annotations(
    session: SessionInfo, cvat_root: Path, *, sampling_rate: float = DEFAULT_SAMPLING_RATE
) -> tuple[pd.DataFrame, str]:
    """Load annotations with a backup and describe their source."""

    if session.annotation_csv.exists():
        backup_path = _backup_annotation_source(session.annotation_csv, session)
        frame = pd.read_csv(session.annotation_csv)
        source = (
            f"Existing FIF-directory annotations ({session.annotation_csv.name}); "
            f"backup saved to {backup_path.name}"
        )
        return frame, source

    frame, source_path = import_annotations(session, cvat_root, sampling_rate=sampling_rate)
    backup_path = _backup_annotation_source(source_path, session)
    frame.to_csv(session.annotation_csv, index=False)
    logger.info("Imported annotations for %s/%s", session.subject_id, session.session_name)
    source = (
        f"CVAT ZIP import ({source_path.name}); backup saved to {backup_path.name}"
    )
    return frame, source
