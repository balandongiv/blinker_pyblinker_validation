"""Interactive CLI for segment-wise MNE annotation management.

Workflow overview (high level flowchart in text):

1. Discover FIF candidates
   → If explicit paths are provided, filter to existing files.
   → Otherwise, search the ``--root`` directory for ``*.fif``.
   → If no files remain, abort with an error.

2. Choose target recording
   → Prompt the user to pick a file (auto-select when only one is available).
   → Derive the companion CSV path (same stem, ``.csv`` extension).

3. Embed existing annotations
   → Load the CSV if present; otherwise start with an empty frame.
   → Normalize the columns to onset/duration/description.
   → Attach the annotations to the loaded ``Raw`` object.

4. Decide on time window
   → Compute total duration and annotation count.
   → If ``--t-start/--t-end`` are provided, use them directly.
   → Else, force the user to enter a window when the annotation count exceeds
     the configured threshold; otherwise default to the whole recording.
   → Flag whether the selected window overlaps existing annotations to inform
     the UI title ("already annotated" vs "new segment").

5. Launch the MNE browser
   → Crop the ``Raw`` to the chosen window.
   → Display the segment with a title showing filename, time range, and status.
   → Block until the user finishes manual edits, then collect updated
     annotations and shift onsets back to absolute time.

6. Merge and persist
   → Ask the user to confirm saving.
   → Remove any existing annotations that overlap the edited window, then merge
     the new ones and sort by onset.
   → Create a timestamped backup in ``backups/`` if a CSV already exists.
   → Write the merged frame back to the main CSV as the latest state.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import mne
import pandas as pd

ANNOTATION_COLUMNS = ["onset", "duration", "description"]


@dataclass
class AnnotationSession:
    """Configuration for a single annotation run."""

    fif_path: Path
    csv_path: Path
    start: float
    end: float
    annotated_before: bool


def find_fif_files(root: Path, provided: Iterable[str] | None = None) -> list[Path]:
    """Return available FIF files from an explicit list or a directory search."""

    if provided:
        paths = [Path(p) for p in provided]
    else:
        paths = sorted(root.glob("*.fif"))
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No FIF files were found in the specified inputs.")
    return existing


def load_annotation_frame(csv_path: Path) -> pd.DataFrame:
    """Load annotations from CSV or return an empty, normalised frame."""

    if not csv_path.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    frame = pd.read_csv(csv_path)
    for column in ANNOTATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[ANNOTATION_COLUMNS].copy()


def annotations_from_frame(frame: pd.DataFrame) -> mne.Annotations:
    """Convert a dataframe into an :class:`mne.Annotations` object."""

    if frame.empty:
        return mne.Annotations(onset=[], duration=[], description=[])
    return mne.Annotations(
        onset=frame["onset"].to_numpy(float),
        duration=frame["duration"].fillna(0).to_numpy(float),
        description=frame["description"].fillna("").astype(str).tolist(),
    )


def frame_from_annotations(annotations: mne.Annotations, offset: float) -> pd.DataFrame:
    """Return a dataframe from ``annotations`` with a time offset applied."""

    return pd.DataFrame(
        {
            "onset": annotations.onset + offset,
            "duration": annotations.duration,
            "description": annotations.description,
        }
    )


def segment_already_annotated(frame: pd.DataFrame, start: float, end: float) -> bool:
    """Determine whether any annotations overlap a time window."""

    if frame.empty:
        return False
    durations = frame["duration"].fillna(0)
    overlap = (frame["onset"] < end) & ((frame["onset"] + durations) > start)
    return bool(overlap.any())


def prompt_time_segment(total_duration: float, required: bool) -> tuple[float, float]:
    """Ask the user for a time window in seconds."""

    prompt_text = (
        "Enter start and end times in seconds (e.g. '0 600'). "
        "Press Enter for the full recording." if not required else ""
    )
    while True:
        raw = input(f"Time window [0-{total_duration:.1f}]: {prompt_text}\n> ").strip()
        if not raw and not required:
            return 0.0, float(total_duration)
        try:
            start_str, end_str = raw.split()
            start, end = float(start_str), float(end_str)
        except ValueError:
            print("Please provide two numbers: start end (in seconds).")
            continue
        if start < 0 or end <= start or end > total_duration:
            print("Invalid range; ensure 0 <= start < end <= total duration.")
            continue
        return start, end


def select_file(fif_files: list[Path]) -> Path:
    """Prompt the user to choose a FIF file from the available options."""

    if len(fif_files) == 1:
        return fif_files[0]
    print("Available FIF files:")
    for idx, path in enumerate(fif_files, start=1):
        print(f"[{idx}] {path.name}")
    while True:
        raw = input("Choose a file by number: ").strip()
        if not raw:
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if 1 <= choice <= len(fif_files):
            return fif_files[choice - 1]
        print("Choice out of range; try again.")


def prepare_session(
    fif_path: Path, annotation_threshold: int, start: float | None, end: float | None
) -> AnnotationSession:
    """Load metadata and decide which segment to annotate."""

    csv_path = fif_path.with_suffix(".csv")
    annotation_frame = load_annotation_frame(csv_path)
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    raw.set_annotations(annotations_from_frame(annotation_frame))
    total_duration = float(raw.times[-1])
    annotation_count = len(annotation_frame)

    if start is not None and end is not None:
        segment = (start, end)
    else:
        require_segment = annotation_count > annotation_threshold
        segment = prompt_time_segment(total_duration, required=require_segment)

    start_time, end_time = segment
    annotated_before = segment_already_annotated(annotation_frame, start_time, end_time)
    return AnnotationSession(
        fif_path=fif_path,
        csv_path=csv_path,
        start=start_time,
        end=end_time,
        annotated_before=annotated_before,
    )


def backup_existing_csv(csv_path: Path) -> None:
    """Create a timestamped backup of an existing annotation CSV."""

    if not csv_path.exists():
        return
    backup_dir = csv_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{csv_path.stem}_{timestamp}{csv_path.suffix}"
    shutil.copy2(csv_path, backup_dir / backup_name)


def merge_annotations(
    global_frame: pd.DataFrame, segment_frame: pd.DataFrame, start: float, end: float
) -> pd.DataFrame:
    """Merge segment annotations into the global set, replacing overlaps."""

    durations = global_frame["duration"].fillna(0)
    overlaps = (global_frame["onset"] < end) & ((global_frame["onset"] + durations) > start)
    kept = global_frame.loc[~overlaps]
    merged = pd.concat([kept, segment_frame], ignore_index=True)
    return merged.sort_values("onset").reset_index(drop=True)


def launch_browser_and_collect(raw: mne.io.Raw, session: AnnotationSession) -> pd.DataFrame:
    """Open the MNE browser for the requested segment and return updated annotations."""

    segment = raw.copy().crop(tmin=session.start, tmax=session.end)
    status = "already annotated" if session.annotated_before else "new segment"
    title = (
        f"{session.fif_path.name} – Segment {session.start:.1f}–{session.end:.1f} s "
        f"– status: {status}"
    )
    print(f"Launching browser with title: {title}")
    segment.plot(title=title, block=True)
    return frame_from_annotations(segment.annotations, offset=session.start)


def save_annotations(csv_path: Path, frame: pd.DataFrame) -> None:
    """Persist annotations with a backup of the previous CSV."""

    backup_existing_csv(csv_path)
    frame.to_csv(csv_path, index=False)
    print(f"Saved annotations to {csv_path}")


def run_ui(root: str, annotation_threshold: int, files: Iterable[str] | None, start: float | None, end: float | None) -> None:
    """Run the interactive annotation workflow."""

    base = Path(root)
    fif_files = find_fif_files(base, provided=files)
    fif_path = select_file(fif_files)
    session = prepare_session(
        fif_path=fif_path,
        annotation_threshold=annotation_threshold,
        start=start,
        end=end,
    )

    annotation_frame = load_annotation_frame(session.csv_path)
    raw = mne.io.read_raw_fif(session.fif_path, preload=True)
    raw.set_annotations(annotations_from_frame(annotation_frame))

    print(
        "\n".join(
            [
                f"Selected: {session.fif_path}",
                f"Time range: {session.start:.1f}–{session.end:.1f} s",
                f"Segment status: {'previously annotated' if session.annotated_before else 'new'}",
            ]
        )
    )

    segment_frame = launch_browser_and_collect(raw, session)

    if input("Save merged annotations to CSV? [y/N]: ").strip().lower() != "y":
        print("Changes discarded at user request.")
        return

    merged = merge_annotations(annotation_frame, segment_frame, session.start, session.end)
    save_annotations(session.csv_path, merged)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""

    parser = argparse.ArgumentParser(description="Annotate FIF files in segments.")
    parser.add_argument(
        "--root",
        default=".",
        help="Directory to search for FIF files when an explicit list is not provided.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Explicit FIF file paths. Overrides --root search when provided.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Annotation count threshold before forcing time segmentation.",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        dest="t_start",
        help="Start time in seconds for the segment to annotate.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        dest="t_end",
        help="End time in seconds for the segment to annotate.",
    )
    return parser


def main() -> None:
    """Entry point for the CLI."""

    parser = build_parser()
    args = parser.parse_args()
    run_ui(
        root=args.root,
        annotation_threshold=args.threshold,
        files=args.files,
        start=args.t_start,
        end=args.t_end,
    )


if __name__ == "__main__":
    main()
