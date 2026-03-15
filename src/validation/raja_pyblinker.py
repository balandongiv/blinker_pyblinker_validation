"""Shared helpers for Raja segment-level PyBlinker processing and comparison."""

from __future__ import annotations

import json
import logging
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import mne
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYBLINKER_REPO_ROOT = REPO_ROOT / "pyblinker"
if PYBLINKER_REPO_ROOT.exists() and str(PYBLINKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PYBLINKER_REPO_ROOT))

from pyblinker.utils.evaluation import blink_comparison

from src.validation.blink_compare import extract_events, normalise_events
from src.validation.stat import RecordingComparison, build_overall_summary


LOGGER = logging.getLogger(__name__)

DEFAULT_PROCESSED_ROOT = Path(r"D:\dataset\drowsy_driving_raja_processed")
DEFAULT_HUMAN_ANNOTATION_ROOT = Path(r"D:\dataset\drowsy_driving_raja\human_label_annotation")
SEGMENT_RAW_RELATIVE_PATH = Path("seg_data_raw") / "eeg_eog_raw.fif"
OUTPUT_DIRNAME = "pyblinker_blinker_validation"
OUTPUT_SUBDIR = "eeg_"
PYBLINKER_RESULT_FILENAME = "pyblinker_results.pkl"
PYBLINKER_METADATA_FILENAME = "pyblinker_results.json"
ANNOTATION_FILENAME = "ear_eog.csv"
COMPARISON_DIFF_FILENAME = "pyblinker_human_diff.csv"
COMPARISON_METRICS_FILENAME = "pyblinker_human_metrics.json"
KNOWN_NON_SEGMENT_DIRS = {
    OUTPUT_DIRNAME,
    "blinker_pyblinker_validation",
    "__pycache__",
}


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """Describe one Raja segment directory."""

    subject_id: str
    segment_id: str
    segment_dir: Path


@dataclass(slots=True)
class SegmentStatus:
    """Track one processing or comparison outcome."""

    subject_id: str
    segment_id: str
    status: str
    reason: str = ""
    fif_path: Path | None = None
    output_path: Path | None = None
    annotation_path: Path | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            "subject_id": self.subject_id,
            "segment_id": self.segment_id,
            "status": self.status,
            "reason": self.reason,
            "fif_path": str(self.fif_path) if self.fif_path is not None else "",
            "output_path": str(self.output_path) if self.output_path is not None else "",
            "annotation_path": (
                str(self.annotation_path) if self.annotation_path is not None else ""
            ),
        }


@dataclass(slots=True)
class SegmentComparisonResult:
    """Store comparison artifacts for one segment."""

    segment: SegmentRecord
    py_path: Path
    annotation_path: Path
    sampling_rate_hz: float
    recording_comparison: RecordingComparison
    diff_table: pd.DataFrame


def configure_logging(verbose: bool = False) -> None:
    """Configure a simple process-wide logging format."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _normalise_filename_filter(filter_filename: str | None) -> str | None:
    if filter_filename is None:
        return None
    cleaned = filter_filename.strip()
    if not cleaned:
        return None
    return Path(cleaned).stem


def matches_filters(
    subject_id: str,
    segment_id: str,
    *,
    filter_subject_id: str | None = None,
    filter_filename: str | None = None,
) -> bool:
    """Return ``True`` when a subject/segment pair passes all active filters."""

    if filter_subject_id and subject_id != filter_subject_id:
        return False

    expected_segment_id = _normalise_filename_filter(filter_filename)
    if expected_segment_id and segment_id != expected_segment_id:
        return False

    return True


def discover_subject_dirs(processed_root: Path) -> list[Path]:
    """Return direct child subject directories under ``processed_root``."""

    if not processed_root.exists():
        return []
    return sorted(path for path in processed_root.iterdir() if path.is_dir())


def discover_segments_for_subject(subject_dir: Path) -> list[SegmentRecord]:
    """Return direct child segment directories for one subject."""

    if not subject_dir.exists():
        return []

    segments: list[SegmentRecord] = []
    for candidate in sorted(subject_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name in KNOWN_NON_SEGMENT_DIRS:
            continue
        segments.append(
            SegmentRecord(
                subject_id=subject_dir.name,
                segment_id=candidate.name,
                segment_dir=candidate,
            )
        )
    return segments


def iter_segments(
    processed_root: Path,
    *,
    filter_subject_id: str | None = None,
    filter_filename: str | None = None,
) -> Iterator[SegmentRecord]:
    """Yield filtered Raja segment records in subject/segment sort order."""

    for subject_dir in discover_subject_dirs(processed_root):
        for segment in discover_segments_for_subject(subject_dir):
            if matches_filters(
                segment.subject_id,
                segment.segment_id,
                filter_subject_id=filter_subject_id,
                filter_filename=filter_filename,
            ):
                yield segment


def count_all_segments(processed_root: Path) -> int:
    """Return the unfiltered number of discovered segment directories."""

    return sum(len(discover_segments_for_subject(subject_dir)) for subject_dir in discover_subject_dirs(processed_root))


def resolve_fif_path(segment_dir: Path) -> Path:
    """Return the canonical Raja segment FIF path."""

    return segment_dir / SEGMENT_RAW_RELATIVE_PATH


def resolve_segment_output_dir(segment_dir: Path) -> Path:
    """Return the requested segment-local PyBlinker output directory."""

    return segment_dir / OUTPUT_DIRNAME / OUTPUT_SUBDIR


def resolve_pyblinker_output_path(segment_dir: Path) -> Path:
    """Return the canonical Raja segment PyBlinker pickle path."""

    return resolve_segment_output_dir(segment_dir) / PYBLINKER_RESULT_FILENAME


def resolve_pyblinker_metadata_path(segment_dir: Path) -> Path:
    """Return the companion JSON metadata path for one segment."""

    return resolve_segment_output_dir(segment_dir) / PYBLINKER_METADATA_FILENAME


def resolve_annotation_csv_path(
    human_annotation_root: Path,
    subject_id: str,
    segment_id: str,
) -> Path:
    """Return the expected human-annotation CSV path."""

    return human_annotation_root / subject_id / segment_id / ANNOTATION_FILENAME


def resolve_segment_diff_path(segment_dir: Path) -> Path:
    """Return the per-segment diff CSV output path."""

    return resolve_segment_output_dir(segment_dir) / COMPARISON_DIFF_FILENAME


def resolve_segment_metrics_path(segment_dir: Path) -> Path:
    """Return the per-segment comparison metrics JSON path."""

    return resolve_segment_output_dir(segment_dir) / COMPARISON_METRICS_FILENAME


def candidate_pyblinker_output_paths(segment_dir: Path) -> tuple[Path, ...]:
    """Return canonical and legacy output paths in lookup order."""

    return (
        resolve_pyblinker_output_path(segment_dir),
        segment_dir / OUTPUT_DIRNAME / PYBLINKER_RESULT_FILENAME,
        segment_dir / "seg_data_raw" / PYBLINKER_RESULT_FILENAME,
    )


def resolve_existing_pyblinker_output_path(segment_dir: Path) -> Path | None:
    """Return the first existing canonical/legacy PyBlinker output path."""

    for candidate in candidate_pyblinker_output_paths(segment_dir):
        if candidate.exists():
            return candidate
    return None


def load_pickle(path: Path):
    """Load a pickle payload."""

    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, payload: Any) -> None:
    """Persist ``payload`` to ``path`` using the highest pickle protocol."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf8",
    )


def _coerce_rate(candidate: Any) -> float | None:
    if candidate is None:
        return None
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def read_raw_sampling_rate(fif_path: Path) -> float:
    """Return the sampling rate of one FIF recording."""

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    return float(raw.info["sfreq"])


def get_payload_sampling_rate(
    payload: Mapping[str, Any],
    *,
    fif_path: Path | None = None,
) -> float:
    """Resolve the sampling rate associated with a PyBlinker payload."""

    metrics = payload.get("metrics", {})
    if isinstance(metrics, Mapping):
        for key in ("sampling_rate_hz", "processed_sampling_rate_hz"):
            rate = _coerce_rate(metrics.get(key))
            if rate is not None:
                return rate

    params = payload.get("params", {})
    if isinstance(params, Mapping):
        rate = _coerce_rate(params.get("resample_rate"))
        if rate is not None:
            return rate

    if fif_path is not None and fif_path.exists():
        return read_raw_sampling_rate(fif_path)

    raise KeyError("Unable to resolve a sampling rate from the PyBlinker payload.")


def get_payload_channel(payload: Mapping[str, Any]) -> str:
    """Return the representative channel name from a PyBlinker payload."""

    metrics = payload.get("metrics", {})
    if isinstance(metrics, Mapping):
        channel = metrics.get("channel")
        if isinstance(channel, str) and channel:
            return channel
    raise KeyError("PyBlinker payload does not expose metrics.channel")


def get_payload_filter_bounds(payload: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Return the detector filter bounds stored in the payload."""

    params = payload.get("params", {})
    if not isinstance(params, Mapping):
        return None, None
    return _coerce_rate(params.get("filter_low")), _coerce_rate(params.get("filter_high"))


def reconstruct_representative_signal(
    fif_path: Path,
    *,
    channel: str,
    sampling_rate_hz: float,
    filter_low: float | None = None,
    filter_high: float | None = None,
) -> np.ndarray:
    """Rebuild a filtered/resampled channel signal when older payloads lack one."""

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    if channel not in raw.ch_names:
        raise ValueError(f"Channel {channel!r} was not found in {fif_path}")

    raw.pick([channel])
    if filter_low is not None or filter_high is not None:
        raw.filter(
            l_freq=filter_low,
            h_freq=filter_high,
            verbose="ERROR",
            n_jobs=1,
        )

    if not math.isclose(float(raw.info["sfreq"]), float(sampling_rate_hz), rel_tol=0.0, abs_tol=1e-9):
        raw.resample(float(sampling_rate_hz), npad="auto", n_jobs=1, verbose="ERROR")

    return raw.get_data()[0].astype(np.float32, copy=False)


def get_payload_signal(
    payload: Mapping[str, Any],
    *,
    fif_path: Path | None = None,
) -> np.ndarray:
    """Return the representative channel signal used for event comparison."""

    for key in ("representative_signal", "signal", "detected_signal"):
        if key in payload:
            signal = np.asarray(payload[key], dtype=np.float32)
            if signal.ndim != 1:
                raise ValueError(f"Expected a 1-D signal for payload key {key!r}")
            if signal.size:
                return signal

    if fif_path is None or not fif_path.exists():
        raise FileNotFoundError(
            "PyBlinker payload does not include a representative signal and the FIF fallback is unavailable."
        )

    sampling_rate_hz = get_payload_sampling_rate(payload, fif_path=fif_path)
    channel = get_payload_channel(payload)
    filter_low, filter_high = get_payload_filter_bounds(payload)
    LOGGER.info(
        "Reconstructing representative signal from %s (channel=%s, sfreq=%s Hz)",
        fif_path,
        channel,
        sampling_rate_hz,
    )
    return reconstruct_representative_signal(
        fif_path,
        channel=channel,
        sampling_rate_hz=sampling_rate_hz,
        filter_low=filter_low,
        filter_high=filter_high,
    )


def extract_pyblinker_events(
    payload: Mapping[str, Any],
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """Return a normalised event table from one PyBlinker payload."""

    if "comparison_events" in payload:
        frame = pd.DataFrame(payload["comparison_events"])
    else:
        frame = extract_events(payload, fallback_key="blinkDetails")

    return normalise_events(
        frame,
        sample_rate=sampling_rate_hz,
        target_rate=sampling_rate_hz,
    )


def load_annotation_table(csv_path: Path) -> pd.DataFrame:
    """Load and validate a Raja human-annotation CSV."""

    frame = pd.read_csv(csv_path)
    if frame.empty:
        return pd.DataFrame(columns=["onset", "duration", "description"])

    frame = frame.copy()
    frame.columns = frame.columns.str.lower()

    if "onset" not in frame.columns:
        if {"start", "end"}.issubset(frame.columns):
            frame["onset"] = pd.to_numeric(frame["start"], errors="coerce")
            frame["duration"] = (
                pd.to_numeric(frame["end"], errors="coerce")
                - pd.to_numeric(frame["start"], errors="coerce")
            )
        else:
            raise ValueError(
                f"Annotation CSV {csv_path} must contain onset/duration columns."
            )

    if "duration" not in frame.columns:
        frame["duration"] = 0.0

    frame["onset"] = pd.to_numeric(frame["onset"], errors="coerce")
    frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce")
    if "description" not in frame.columns:
        frame["description"] = "Blink"
    frame["description"] = frame["description"].fillna("Blink").astype("string")

    clean = frame.dropna(subset=["onset", "duration"]).copy()
    clean = clean.loc[clean["duration"] >= 0].copy()
    if clean.empty:
        return pd.DataFrame(columns=["onset", "duration", "description"])

    median_onset = float(clean["onset"].median())
    if math.isfinite(median_onset) and median_onset > 10000:
        clean["onset"] = clean["onset"] / 1000.0
        clean["duration"] = clean["duration"] / 1000.0

    clean = clean.sort_values(["onset", "duration"], kind="mergesort").reset_index(drop=True)
    return clean.loc[:, ["onset", "duration", "description"]]


def annotation_table_to_events(
    annotation_table: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """Convert annotation onsets/durations into 1-based sample intervals."""

    if annotation_table.empty:
        return pd.DataFrame(columns=["start_blink", "end_blink"], dtype=int)

    starts = np.rint(annotation_table["onset"].to_numpy(dtype=float) * sampling_rate_hz).astype(int)
    ends = np.rint(
        (annotation_table["onset"].to_numpy(dtype=float) + annotation_table["duration"].to_numpy(dtype=float))
        * sampling_rate_hz
    ).astype(int)
    ends = np.maximum(ends, starts)

    events = pd.DataFrame({"start_blink": starts, "end_blink": ends})
    columns = ["start_blink", "end_blink"]
    if (events[columns] < 0).any().any():
        raise ValueError("Annotation events must not contain negative sample indices.")
    if (events[columns] == 0).any().any():
        events[columns] = events[columns] + 1
    return events.sort_values("start_blink", kind="mergesort").reset_index(drop=True)


def run_pyblinker_on_segment(
    segment: SegmentRecord,
    *,
    overwrite: bool = False,
    filter_low: float = 0.5,
    filter_high: float = 30.0,
    resample_rate: float = 200.0,
    n_jobs: int = 1,
    use_multiprocessing: bool = True,
) -> SegmentStatus:
    """Run PyBlinker on one Raja segment and save a rich pickle payload."""

    fif_path = resolve_fif_path(segment.segment_dir)
    output_path = resolve_pyblinker_output_path(segment.segment_dir)

    if not fif_path.exists():
        return SegmentStatus(
            subject_id=segment.subject_id,
            segment_id=segment.segment_id,
            status="skipped_missing_fif",
            reason=f"Missing input FIF: {fif_path}",
            fif_path=fif_path,
            output_path=output_path,
        )

    if output_path.exists() and not overwrite:
        return SegmentStatus(
            subject_id=segment.subject_id,
            segment_id=segment.segment_id,
            status="skipped_existing_output",
            reason=f"Output already exists: {output_path}",
            fif_path=fif_path,
            output_path=output_path,
        )

    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    except Exception as exc:
        return SegmentStatus(
            subject_id=segment.subject_id,
            segment_id=segment.segment_id,
            status="failed_load_fif",
            reason=str(exc),
            fif_path=fif_path,
            output_path=output_path,
        )

    from pyblinker.blinker.pyblinker import BlinkDetector

    try:
        detector = BlinkDetector(
            raw.copy(),
            visualize=False,
            annot_label="eye_blink",
            filter_low=filter_low,
            filter_high=filter_high,
            resample_rate=float(resample_rate),
            n_jobs=n_jobs,
            use_multiprocessing=use_multiprocessing,
        )
        annotations, channel, n_good, blink_details, _fig_data, selected_channel = detector.get_blink()
    except Exception as exc:
        return SegmentStatus(
            subject_id=segment.subject_id,
            segment_id=segment.segment_id,
            status="failed_pyblinker",
            reason=str(exc),
            fif_path=fif_path,
            output_path=output_path,
        )

    events_frame = pd.DataFrame(blink_details).reset_index(drop=True)
    processed_raw = detector.raw_data
    processed_rate = float(processed_raw.info["sfreq"])
    representative_signal = processed_raw.get_data(picks=[channel])[0].astype(np.float32, copy=False)
    comparison_events = normalise_events(
        events_frame,
        sample_rate=processed_rate,
        target_rate=processed_rate,
    )

    payload = {
        "events": events_frame,
        "comparison_events": comparison_events,
        "annotations": annotations,
        "selected_channel": selected_channel,
        "representative_signal": representative_signal,
        "metrics": {
            "subject_id": segment.subject_id,
            "segment_id": segment.segment_id,
            "channel": str(channel),
            "n_good_blinks": int(n_good),
            "n_events": int(len(comparison_events)),
            "sampling_rate_hz": processed_rate,
            "raw_sampling_rate_hz": float(raw.info["sfreq"]),
            "input_file": str(fif_path),
        },
        "params": {
            "filter_low": float(filter_low),
            "filter_high": float(filter_high),
            "resample_rate": float(resample_rate),
            "n_jobs": int(n_jobs),
            "use_multiprocessing": bool(use_multiprocessing),
        },
    }

    try:
        save_pickle(output_path, payload)
        write_json(
            resolve_pyblinker_metadata_path(segment.segment_dir),
            {
                "subject_id": segment.subject_id,
                "segment_id": segment.segment_id,
                "output_path": str(output_path),
                "metrics": payload["metrics"],
                "params": payload["params"],
            },
        )
    except Exception as exc:
        return SegmentStatus(
            subject_id=segment.subject_id,
            segment_id=segment.segment_id,
            status="failed_save_output",
            reason=str(exc),
            fif_path=fif_path,
            output_path=output_path,
        )

    return SegmentStatus(
        subject_id=segment.subject_id,
        segment_id=segment.segment_id,
        status="processed",
        reason=f"Saved {len(comparison_events)} events to {output_path}",
        fif_path=fif_path,
        output_path=output_path,
    )


def compare_segment_with_human_annotations(
    segment: SegmentRecord,
    *,
    human_annotation_root: Path,
    tolerance_samples: int,
) -> tuple[SegmentStatus, SegmentComparisonResult | None]:
    """Compare one PyBlinker segment output against Raja human annotations."""

    fif_path = resolve_fif_path(segment.segment_dir)
    py_path = resolve_existing_pyblinker_output_path(segment.segment_dir)
    if py_path is None:
        return (
            SegmentStatus(
                subject_id=segment.subject_id,
                segment_id=segment.segment_id,
                status="skipped_missing_pyblinker",
                reason="No pyblinker_results.pkl found in canonical or legacy locations.",
                fif_path=fif_path,
            ),
            None,
        )

    annotation_path = resolve_annotation_csv_path(
        human_annotation_root,
        segment.subject_id,
        segment.segment_id,
    )
    if not annotation_path.exists():
        return (
            SegmentStatus(
                subject_id=segment.subject_id,
                segment_id=segment.segment_id,
                status="skipped_missing_annotation",
                reason=f"Missing human annotation CSV: {annotation_path}",
                fif_path=fif_path,
                output_path=py_path,
                annotation_path=annotation_path,
            ),
            None,
        )

    try:
        payload = load_pickle(py_path)
    except Exception as exc:
        return (
            SegmentStatus(
                subject_id=segment.subject_id,
                segment_id=segment.segment_id,
                status="failed_load_pyblinker",
                reason=str(exc),
                fif_path=fif_path,
                output_path=py_path,
                annotation_path=annotation_path,
            ),
            None,
        )

    try:
        annotation_table = load_annotation_table(annotation_path)
    except Exception as exc:
        return (
            SegmentStatus(
                subject_id=segment.subject_id,
                segment_id=segment.segment_id,
                status="failed_load_annotation",
                reason=str(exc),
                fif_path=fif_path,
                output_path=py_path,
                annotation_path=annotation_path,
            ),
            None,
        )

    try:
        sampling_rate_hz = get_payload_sampling_rate(payload, fif_path=fif_path if fif_path.exists() else None)
        py_events = extract_pyblinker_events(payload, sampling_rate_hz=sampling_rate_hz)
        ground_truth_events = annotation_table_to_events(
            annotation_table,
            sampling_rate_hz=sampling_rate_hz,
        )
        detected_signal = get_payload_signal(payload, fif_path=fif_path if fif_path.exists() else None)
    except Exception as exc:
        return (
            SegmentStatus(
                subject_id=segment.subject_id,
                segment_id=segment.segment_id,
                status="failed_prepare_comparison",
                reason=str(exc),
                fif_path=fif_path,
                output_path=py_path,
                annotation_path=annotation_path,
            ),
            None,
        )

    try:
        comparison = blink_comparison.compare_detected_vs_ground_truth(
            py_events,
            ground_truth_events,
            float(sampling_rate_hz),
            tolerance_samples=tolerance_samples,
            n_preview_rows=10,
            n_diff_rows=max(50, len(py_events) + len(ground_truth_events)),
            detected_signal=detected_signal,
        )
    except Exception as exc:
        return (
            SegmentStatus(
                subject_id=segment.subject_id,
                segment_id=segment.segment_id,
                status="failed_comparison",
                reason=str(exc),
                fif_path=fif_path,
                output_path=py_path,
                annotation_path=annotation_path,
            ),
            None,
        )

    recording_id = f"{segment.subject_id}/{segment.segment_id}"
    recording_comparison = RecordingComparison(
        recording_id=recording_id,
        py_events=py_events,
        blinker_events=ground_truth_events,
        metrics=comparison.metrics,
    )
    result = SegmentComparisonResult(
        segment=segment,
        py_path=py_path,
        annotation_path=annotation_path,
        sampling_rate_hz=float(sampling_rate_hz),
        recording_comparison=recording_comparison,
        diff_table=comparison.diff_table,
    )
    status = SegmentStatus(
        subject_id=segment.subject_id,
        segment_id=segment.segment_id,
        status="compared",
        reason=(
            "Compared PyBlinker output against human annotations "
            f"at {sampling_rate_hz:.3f} Hz"
        ),
        fif_path=fif_path,
        output_path=py_path,
        annotation_path=annotation_path,
    )
    return status, result


def write_segment_comparison_outputs(result: SegmentComparisonResult) -> None:
    """Persist per-segment comparison artifacts beside the PyBlinker output."""

    diff_path = resolve_segment_diff_path(result.segment.segment_dir)
    metrics_path = resolve_segment_metrics_path(result.segment.segment_dir)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    result.diff_table.to_csv(diff_path, index=False)
    write_json(
        metrics_path,
        {
            "subject_id": result.segment.subject_id,
            "segment_id": result.segment.segment_id,
            "recording_id": result.recording_comparison.recording_id,
            "sampling_rate_hz": result.sampling_rate_hz,
            "py_path": result.py_path,
            "annotation_path": result.annotation_path,
            "metrics": dict(result.recording_comparison.metrics),
        },
    )


def status_records_to_frame(statuses: Iterable[SegmentStatus]) -> pd.DataFrame:
    """Convert status records into a stable tabular form."""

    rows = [status.to_dict() for status in statuses]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "segment_id",
                "status",
                "reason",
                "fif_path",
                "output_path",
                "annotation_path",
            ]
        )
    return frame.sort_values(["subject_id", "segment_id"], kind="mergesort").reset_index(drop=True)


def build_subject_summary_frame(segment_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment metrics into one summary row per subject."""

    if segment_summary.empty or "subject_id" not in segment_summary.columns:
        return pd.DataFrame()

    subject_rows: list[dict[str, Any]] = []
    for subject_id, subject_frame in segment_summary.groupby("subject_id", sort=True):
        overall = build_overall_summary(subject_frame)
        row = overall.to_dict()
        row["subject_id"] = subject_id
        row["segment_count"] = int(len(subject_frame))
        subject_rows.append(row)

    if not subject_rows:
        return pd.DataFrame()

    return pd.DataFrame(subject_rows).sort_values("subject_id", kind="mergesort").reset_index(drop=True)
