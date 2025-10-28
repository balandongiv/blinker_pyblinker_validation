"""Simple script for annotating and plotting blink intervals from PyBlinker and Blinker."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration (easy to edit for quick EDA)
# -----------------------------------------------------------------------------
ASSUMED_FRAME_RATE = 100.0
USE_EEG_SAMPLING = False
PLOT_DURATION = 20.0
PLOT_START = 0.0
CLIP_TO_BOUNDS = False

# Resolve default data paths relative to the repository root so the script can be
# launched from any working directory (e.g., IDEs may default to the project or
# tutorial folder).  This avoids FileNotFoundError when the relative paths below
# are interpreted from an unexpected CWD.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
BLINKER_PKL = _REPO_ROOT / "data_test/S01_20170519_043933_3/blinker/blinkFits.pkl"
PYBLINKER_PKL = _REPO_ROOT / "data_test/S01_20170519_043933_3/pyblinker/blink_details.pkl"
FIF_PATH = _REPO_ROOT / "data_test/S01_20170519_043933_3/seg_annotated_raw.fif"
OUT_FIF = FIF_PATH.with_name(FIF_PATH.stem + "_with_blinks.fif")


@dataclass
class AnnotationSummary:
    loaded: int
    after_na: int
    dropped_na: int


class IntervalError(Exception):
    """Custom error raised when an interval cannot be converted."""


def annotate_and_plot_blinks(
    blinker_pkl: Path | str,
    pyblinker_pkl: Path | str,
    fif_path: Path | str,
    out_fif: Path | str | None = None,
    assumed_frame_rate: float = ASSUMED_FRAME_RATE,
    use_eeg_sampling: bool = USE_EEG_SAMPLING,
    clip_to_bounds: bool = CLIP_TO_BOUNDS,
    plot_duration: float | None = PLOT_DURATION,
    start: float | None = PLOT_START,
    verbose: bool = True,
) -> Path:
    """Summary
    Load blink intervals from PyBlinker and Blinker, convert them to MNE Annotations using either an assumed frame rate (default 100 Hz) or the EEG sampling rate, attach to a FIF file, save a copy, and plot for quick visual cross-checking.

    Parameters

    * `blinker_pkl` (path-like): Path to Blinker `blinkFits.pkl`.
    * `pyblinker_pkl` (path-like): Path to PyBlinker `blink_details.pkl`.
    * `fif_path` (path-like): Path to input FIF file.
    * `out_fif` (path-like, optional): Output path; defaults to input filename with `_with_blinks` suffix.
    * `assumed_frame_rate` (float, default `100.0`): Frame rate (Hz) used if `use_eeg_sampling=False`; time = frame / rate.
    * `use_eeg_sampling` (bool, default `False`): If `True`, interpret indices as EEG samples and convert using `raw.info['sfreq']`.
    * `clip_to_bounds` (bool, default `False`): If `True`, clip out-of-range intervals to `[0, raw.times[-1]]`; else skip with a warning.
    * `plot_duration` (float, optional): Initial window length (seconds) for plotting.
    * `start` (float, optional): Initial start time (seconds) for plotting.
    * `verbose` (bool, default `True`): Print EDA summaries and warnings.

    Data Requirements

    * PyBlinker DataFrame must contain `left_zero`, `right_zero` (ints). Optional: `start_blink`, `end_blink`, `max_blink`.
    * Blinker DataFrame must contain `leftZero`, `rightZero` (ints). Optional: `maxFrame`.
    * Indices must be non-negative integers; NaNs in required columns are dropped.

    Behavior

    * Uses `assumed_frame_rate` when `use_eeg_sampling=False` (default EDA mode for unknown rates).
    * Builds `"blink_pyblinker"` and `"blink_blinker"` annotation tracks.
    * Preserves existing annotations by concatenation.
    * Saves an annotated copy and opens an interactive plot for visual comparison.

    Returns

    * `out_fif` (path-like): Path to the saved annotated FIF file.

    Raises

    * `FileNotFoundError`: If any input path is missing.
    * `ValueError`: Missing required columns or invalid indices.
    * `RuntimeError`: Save or plotting failures.

    Notes

    * If the pickle indices are not in the same clock domain as the EEG, set `use_eeg_sampling=True` to interpret them as EEG samples; otherwise keep the default and tune `assumed_frame_rate` (100 Hz by default).
    * Overlapping intervals are expected and shown as overlapping spans in the MNE plot.

    Example Usage (prose)

    * Default EDA with 100 Hz assumption: set `assumed_frame_rate=100.0`, `use_eeg_sampling=False`, run the function; a new FIF with both blink tracks is saved and plotted.
    * If later you learn the tool wrote EEG sample indices: set `use_eeg_sampling=True` (ignores `assumed_frame_rate`) and rerun.
    * To focus on the first 30 s of data: set `plot_duration=30.0`, `start=0.0`.
    """

    blinker_path = Path(blinker_pkl)
    pyblinker_path = Path(pyblinker_pkl)
    fif_path = Path(fif_path)
    out_path = Path(out_fif) if out_fif is not None else _derive_out_path(fif_path)

    for path in (blinker_path, pyblinker_path, fif_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose="ERROR")

    py_df, py_summary = _load_blink_dataframe(
        pyblinker_path,
        required_columns=("left_zero", "right_zero"),
        label="PyBlinker",
        verbose=verbose,
    )
    blinker_df, blinker_summary = _load_blink_dataframe(
        blinker_path,
        required_columns=("leftZero", "rightZero"),
        label="Blinker",
        verbose=verbose,
    )

    to_seconds = _build_time_mapper(
        raw=raw,
        assumed_frame_rate=assumed_frame_rate,
        use_eeg_sampling=use_eeg_sampling,
    )

    if verbose:
        effective_rate = raw.info["sfreq"] if use_eeg_sampling else assumed_frame_rate
        mapping_source = "raw.info['sfreq']" if use_eeg_sampling else "ASSUMED_FRAME_RATE"
        print(
            f"Using {mapping_source} = {effective_rate:.3f} Hz for frame-to-second conversion"
        )
        print(
            f"Recording duration: {raw.times[-1]:.3f} s | Existing annotations: {len(raw.annotations)}"
        )

    raw_end = raw.times[-1]
    base_orig_time = raw.annotations.orig_time
    time_offset = raw.first_time if base_orig_time is not None else 0.0

    py_annotations, py_kept = _build_annotations(
        df=py_df,
        left_col="left_zero",
        right_col="right_zero",
        description="blink_pyblinker",
        to_seconds=to_seconds,
        raw_end=raw_end,
        clip_to_bounds=clip_to_bounds,
        verbose=verbose,
        orig_time=base_orig_time,
        time_offset=time_offset,
    )
    blinker_annotations, blinker_kept = _build_annotations(
        df=blinker_df,
        left_col="leftZero",
        right_col="rightZero",
        description="blink_blinker",
        to_seconds=to_seconds,
        raw_end=raw_end,
        clip_to_bounds=clip_to_bounds,
        verbose=verbose,
        orig_time=base_orig_time,
        time_offset=time_offset,
    )

    if verbose:
        _print_summary("PyBlinker", py_summary, py_kept)
        _print_summary("Blinker", blinker_summary, blinker_kept)
        _print_sample_intervals("PyBlinker", py_kept)
        _print_sample_intervals("Blinker", blinker_kept)
        _print_alignment_stats(py_annotations, blinker_annotations)

    merged_annotations = _merge_annotations(
        raw,
        additions=[ann for ann in (py_annotations, blinker_annotations) if ann is not None],
    )

    raw.set_annotations(merged_annotations)

    try:
        raw.save(str(out_path), overwrite=True)
    except Exception as exc:  # pragma: no cover - best effort logging
        raise RuntimeError(f"Failed to save annotated FIF to {out_path}: {exc}") from exc

    if verbose:
        print(f"Saved annotated FIF: {out_path}")
        if raw.get_channel_types(picks="eog"):
            print("Tip: Use raw.plot(picks='eog') to focus on eye channels.")

    try:
        raw.plot(
            scalings="auto",
            duration=plot_duration,
            start=start,
            title="Blink overlay: pyblinker vs blinker",
        )
    except Exception as exc:  # pragma: no cover - interactive behavior
        raise RuntimeError(f"Failed to plot raw data with annotations: {exc}") from exc

    return out_path


def _derive_out_path(fif_path: Path) -> Path:
    return fif_path.with_name(f"{fif_path.stem}_with_blinks{fif_path.suffix}")


def _load_blink_dataframe(
    path: Path,
    required_columns: Sequence[str],
    label: str,
    verbose: bool,
) -> Tuple[pd.DataFrame, AnnotationSummary]:
    try:
        df = pd.read_pickle(path)
    except Exception as exc:  # pragma: no cover - IO handling
        raise RuntimeError(f"Failed to read {label} pickle at {path}: {exc}") from exc

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} data missing required columns {missing}; available: {list(df.columns)}"
        )

    loaded = len(df)
    df = df.dropna(subset=required_columns)
    after_drop = len(df)
    dropped = loaded - after_drop

    if verbose and dropped:
        print(f"{label}: dropped {dropped} rows with NaNs in required columns")

    summary = AnnotationSummary(loaded=loaded, after_na=after_drop, dropped_na=dropped)
    return df, summary


def _build_time_mapper(
    raw: mne.io.BaseRaw,
    assumed_frame_rate: float,
    use_eeg_sampling: bool,
) -> Callable[[float], float]:
    if use_eeg_sampling:
        sfreq = float(raw.info["sfreq"])
        if sfreq <= 0:
            raise ValueError("EEG sampling frequency must be positive")
        return lambda x: float(x) / sfreq
    if assumed_frame_rate <= 0:
        raise ValueError("assumed_frame_rate must be positive")
    return lambda x: float(x) / float(assumed_frame_rate)


def _validate_index(value: float, idx: int, label: str) -> int:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        raise IntervalError(f"{label}: non-finite index at row {idx}")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise IntervalError(f"{label}: non-numeric index at row {idx}: {value}") from None
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-6:
        raise IntervalError(f"{label}: non-integer index at row {idx}: {value}")
    if rounded < 0:
        raise IntervalError(f"{label}: negative index at row {idx}: {value}")
    return rounded


def _build_annotations(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    description: str,
    to_seconds: Callable[[float], float],
    raw_end: float,
    clip_to_bounds: bool,
    verbose: bool,
    orig_time: object | None,
    time_offset: float = 0.0,
) -> Tuple[mne.Annotations | None, List[Tuple[float, float]]]:
    if df.empty:
        if verbose:
            print(f"{description}: no intervals after filtering")
        return None, []

    onsets: List[float] = []
    durations: List[float] = []
    descriptions: List[str] = []
    kept_intervals: List[Tuple[float, float]] = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            left_idx = _validate_index(row[left_col], idx, description)
            right_idx = _validate_index(row[right_col], idx, description)
        except IntervalError as exc:
            skipped += 1
            if verbose:
                print(f"Skipping {description} row {idx}: {exc}")
            continue

        onset = to_seconds(left_idx)
        offset = to_seconds(right_idx)

        if offset < onset:
            skipped += 1
            if verbose:
                print(
                    f"Skipping {description} row {idx}: right index precedes left ({left_idx} > {right_idx})"
                )
            continue

        if clip_to_bounds:
            onset = max(0.0, onset)
            offset = min(raw_end, offset)
        else:
            if onset < 0 or offset > raw_end:
                skipped += 1
                if verbose:
                    print(
                        f"Skipping {description} row {idx}: interval [{onset:.3f}, {offset:.3f}] outside raw bounds"
                    )
                continue

        duration = max(0.0, offset - onset)
        if duration <= 0:
            skipped += 1
            if verbose:
                print(
                    f"Skipping {description} row {idx}: non-positive duration after checks ({duration:.6f})"
                )
            continue

        adjusted_onset = onset + (time_offset if orig_time is not None else 0.0)
        onsets.append(adjusted_onset)
        durations.append(duration)
        descriptions.append(description)
        kept_intervals.append((onset, offset))

    annotations = None
    if onsets:
        annotations = mne.Annotations(
            onsets,
            durations,
            descriptions,
            orig_time=orig_time,
        )

    if verbose:
        print(
            f"{description}: kept {len(onsets)} intervals, skipped {skipped} (total {len(df)})"
        )

    return annotations, kept_intervals


def _merge_annotations(
    raw: mne.io.BaseRaw,
    additions: Sequence[mne.Annotations],
) -> mne.Annotations:
    base = raw.annotations
    if not additions:
        return base.copy()

    target_orig_time = base.orig_time
    merged_onsets = base.onset.tolist()
    merged_durations = base.duration.tolist()
    merged_descriptions = base.description.tolist()
    raw_end = float(raw.times[-1])
    first_time = float(raw.first_time or 0.0)

    for annotation in additions:
        aligned = _align_annotation_origin(
            annotation,
            target_orig_time,
            first_time=first_time,
            raw_end=raw_end,
        )
        if aligned is None:
            continue

        if aligned.orig_time != target_orig_time:
            existing = mne.Annotations(
                merged_onsets,
                merged_durations,
                merged_descriptions,
                orig_time=target_orig_time,
            )
            existing_aligned = _align_annotation_origin(
                existing,
                aligned.orig_time,
                first_time=first_time,
                raw_end=raw_end,
            )
            target_orig_time = aligned.orig_time
            merged_onsets = existing_aligned.onset.tolist()
            merged_durations = existing_aligned.duration.tolist()
            merged_descriptions = existing_aligned.description.tolist()

        merged_onsets.extend(aligned.onset.tolist())
        merged_durations.extend(aligned.duration.tolist())
        merged_descriptions.extend(aligned.description.tolist())

    return mne.Annotations(
        merged_onsets,
        merged_durations,
        merged_descriptions,
        orig_time=target_orig_time,
    )


def _align_annotation_origin(
    annotations: mne.Annotations | None,
    target_orig_time: object | None,
    *,
    first_time: float,
    raw_end: float,
) -> mne.Annotations | None:
    if annotations is None:
        return None

    if len(annotations) == 0:
        return (
            annotations
            if annotations.orig_time == target_orig_time
            else mne.Annotations([], [], [], orig_time=target_orig_time)
        )

    if annotations.orig_time == target_orig_time:
        return annotations

    onsets = np.array(annotations.onset, dtype=float, copy=True)
    durations = np.array(annotations.duration, dtype=float, copy=True)
    descriptions = annotations.description.tolist()
    source_orig_time = annotations.orig_time

    if target_orig_time is None:
        if source_orig_time is None:
            adjusted_onsets = onsets
        else:
            adjusted_onsets = onsets - float(first_time)
        new_orig_time = None
    else:
        if source_orig_time is None:
            if onsets.size and np.nanmax(onsets) <= raw_end + 1.0:
                adjusted_onsets = onsets + float(first_time)
            else:
                adjusted_onsets = onsets
        else:
            source_stamp = _orig_time_to_seconds(source_orig_time)
            target_stamp = _orig_time_to_seconds(target_orig_time)
            if source_stamp is None or target_stamp is None:
                raise RuntimeError(
                    "Unable to align annotations because orig_time values are not convertible to seconds."
                )
            adjusted_onsets = onsets + (source_stamp - target_stamp)
        new_orig_time = target_orig_time

    return mne.Annotations(
        adjusted_onsets,
        durations,
        descriptions,
        orig_time=new_orig_time,
    )


def _orig_time_to_seconds(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        seconds, microseconds = value
        return float(seconds) + float(microseconds) * 1e-6
    try:
        return float(pd.Timestamp(value).timestamp())
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise TypeError(f"Unsupported orig_time value: {value!r}") from exc


def _print_summary(label: str, summary: AnnotationSummary, kept: List[Tuple[float, float]]):
    kept_count = len(kept)
    skipped_after_na = max(summary.after_na - kept_count, 0)
    print(
        f"{label}: loaded {summary.loaded} | kept {kept_count} | dropped_na {summary.dropped_na} | skipped_qc {skipped_after_na}"
    )


def _print_sample_intervals(
    label: str,
    intervals: Sequence[Tuple[float, float]],
    n: int = 3,
) -> None:
    if not intervals:
        print(f"{label}: no intervals to preview")
        return
    onsets = [interval[0] for interval in intervals]
    offsets = [interval[1] for interval in intervals]
    print(f"{label}: first {n} intervals (s):")
    for onset, offset in zip(onsets[:n], offsets[:n]):
        print(f"  [{onset:.3f}, {offset:.3f}]")
    print(f"{label}: last {n} intervals (s):")
    for onset, offset in zip(onsets[-n:], offsets[-n:]):
        print(f"  [{onset:.3f}, {offset:.3f}]")


def _print_alignment_stats(
    py_annotations: mne.Annotations | None,
    blinker_annotations: mne.Annotations | None,
) -> None:
    if py_annotations is None or blinker_annotations is None:
        print("Alignment summary: insufficient data for comparison")
        return

    py_onsets = np.asarray(py_annotations.onset)
    blinker_onsets = np.asarray(blinker_annotations.onset)
    if py_onsets.size == 0 or blinker_onsets.size == 0:
        print("Alignment summary: insufficient data for comparison")
        return

    diffs = []
    for onset in py_onsets:
        idx = np.abs(blinker_onsets - onset).argmin()
        diffs.append(blinker_onsets[idx] - onset)

    diffs = np.asarray(diffs)
    print("Alignment summary (Blinker - PyBlinker onsets in seconds):")
    print(
        f"  median {np.median(diffs):.4f} | mean {np.mean(diffs):.4f} | std {np.std(diffs):.4f} | MAD {np.median(np.abs(diffs)):.4f}"
    )


def _main(argv: Sequence[str]) -> int:
    try:
        annotate_and_plot_blinks(
            blinker_pkl=BLINKER_PKL,
            pyblinker_pkl=PYBLINKER_PKL,
            fif_path=FIF_PATH,
            out_fif=OUT_FIF,
            assumed_frame_rate=ASSUMED_FRAME_RATE,
            use_eeg_sampling=USE_EEG_SAMPLING,
            clip_to_bounds=CLIP_TO_BOUNDS,
            plot_duration=PLOT_DURATION,
            start=PLOT_START,
            verbose=True,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
