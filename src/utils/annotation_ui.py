"""Tkinter GUI for segment-wise MNE annotation management.

Flowchart-style walkthrough of the GUI logic:

1. Launch and discover FIF files
   → Prefer the default dataset root ``D:\dataset\murat_2018`` and traverse
     all nested folders for ``*.fif`` files. An explicit ``--root`` or
     ``--files`` list can override the default search.
   → Populate the list widget; if nothing is found, show an error dialog and
   → exit.

2. Pick a target recording
   → The user clicks a file in the list; the app derives the companion CSV path
     with the same stem and a ``.csv`` extension.
   → Load the CSV (if it exists) into a normalized dataframe and report the
     current annotation count.
   → Fetch the last edit timestamp from the CSV (if present) so the info panel
     can show when annotations were last saved.

3. Inspect metadata
   → Open the FIF header (no preload) to determine total duration.
   → Update the info panel with recording length, the number of existing
     annotations, and the last edit time.

4. Choose a time window
   → The "Start (s)" and "End (s)" fields accept numeric inputs.
   → If both fields are empty **and** the annotation count is below the
     threshold, the app defaults to the full recording.
   → If the annotation count exceeds the threshold, the app requires explicit
     start/end values and will warn until valid numbers are provided.
   → The app checks whether any annotations overlap the chosen window so the
     plot title can communicate "already annotated" vs "new segment".

5. Launch the MNE browser
   → Load the FIF with ``preload=True`` and attach current annotations.
   → Crop the raw object to the requested window and open ``.plot`` with a
     title containing the filename, range, status, and segment times.
   → The plot blocks while the user edits annotations; on close, the updated
     annotations are exported with onsets shifted back to absolute time.

6. Merge and save
   → A confirmation dialog asks whether to merge and persist changes.
   → The app removes any existing annotations that overlap the edited window,
     merges the new ones, and sorts by onset.
   → Before overwriting the main CSV, it writes a timestamped backup to a
     ``backups/`` folder beside the CSV.
   → A logger writes major events (segments processed, additions/removals,
     totals) to a history panel and to a file beside the dataset root.
   → The main CSV always reflects the latest global annotation set for the FIF
     file.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Iterable, Optional

import mne
import pandas as pd
from tkinter import (
    BOTH,
    BOTTOM,
    END,
    LEFT,
    RIGHT,
    TOP,
    X,
    Y,
    Button,
    Entry,
    Frame,
    Label,
    Listbox,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    messagebox,
)

ANNOTATION_COLUMNS = ["onset", "duration", "description"]
DEFAULT_ROOT = Path(r"D:\\dataset\\murat_2018")
logger = logging.getLogger(__name__)


def configure_logger(log_path: Path) -> None:
    """Configure a file logger for annotation activity."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    existing_files = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
    ]
    if not existing_files:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(file_handler)


def read_history(log_path: Path, max_lines: int = 50) -> list[str]:
    """Return the last ``max_lines`` of the history log if it exists."""

    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as log_file:
        lines = log_file.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def last_edit_timestamp(csv_path: Path) -> str:
    """Return the last modification time for ``csv_path`` or ``"None"``."""

    if not csv_path.exists():
        return "None"
    modified = datetime.fromtimestamp(csv_path.stat().st_mtime)
    return modified.strftime("%Y-%m-%d %H:%M:%S")


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
        paths = sorted(root.rglob("*.fif"))
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
    raise RuntimeError("prompt_time_segment is not used in the GUI workflow.")


def select_file(fif_files: list[Path]) -> Path:
    """Prompt the user to choose a FIF file from the available options."""
    raise RuntimeError("select_file is not used in the GUI workflow.")


def prepare_session(
    fif_path: Path, annotation_threshold: int, start: float | None, end: float | None
) -> AnnotationSession:
    """Preserved for CLI compatibility; the GUI does not use this helper."""

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
) -> tuple[pd.DataFrame, int]:
    """Merge segment annotations into the global set, replacing overlaps."""

    durations = global_frame["duration"].fillna(0)
    overlaps = (global_frame["onset"] < end) & ((global_frame["onset"] + durations) > start)
    kept = global_frame.loc[~overlaps]
    merged = pd.concat([kept, segment_frame], ignore_index=True)
    merged = merged.sort_values("onset").reset_index(drop=True)
    return merged, int(overlaps.sum())


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


class AnnotationApp:
    """Tkinter GUI controller for annotation management."""

    def __init__(
        self,
        root_path: Path,
        provided_files: Optional[Iterable[str]] = None,
        annotation_threshold: int = 100,
    ) -> None:
        self.root_path = root_path
        self.provided_files = provided_files
        self.annotation_threshold = annotation_threshold

        self.log_path = (self.root_path if self.root_path.exists() else Path.cwd()) / "annotation_ui.log"
        configure_logger(self.log_path)

        self.window = Tk()
        self.window.title("MNE Annotation Helper")

        self.status_var = StringVar(value="Select a FIF file to begin.")
        self.info_var = StringVar(value="No file selected.")
        self.segment_status_var = StringVar(value="")

        self.start_entry: Entry
        self.end_entry: Entry
        self.file_list: Listbox
        self.history_text: Text

        self.annotation_frame = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        self.total_duration: float | None = None
        self.selected_file: Path | None = None

        self._build_ui()
        self._populate_files()
        self._refresh_history()

    def _build_ui(self) -> None:
        """Construct the Tkinter layout."""

        top_frame = Frame(self.window)
        top_frame.pack(side=TOP, fill=BOTH, expand=True, padx=8, pady=8)

        list_frame = Frame(top_frame)
        list_frame.pack(side=LEFT, fill=BOTH, expand=True)

        Label(list_frame, text="Available FIF files:").pack(anchor="w")
        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.file_list = Listbox(list_frame, yscrollcommand=scrollbar.set, height=12)
        self.file_list.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.file_list.yview)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)

        controls = Frame(top_frame)
        controls.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        Label(controls, textvariable=self.info_var, wraplength=320, justify=LEFT).pack(
            anchor="w", pady=(0, 8)
        )

        entry_frame = Frame(controls)
        entry_frame.pack(fill=X, pady=4)
        Label(entry_frame, text="Start (s):").pack(side=LEFT)
        self.start_entry = Entry(entry_frame, width=10)
        self.start_entry.pack(side=LEFT, padx=(4, 12))
        Label(entry_frame, text="End (s):").pack(side=LEFT)
        self.end_entry = Entry(entry_frame, width=10)
        self.end_entry.pack(side=LEFT, padx=4)

        Button(controls, text="Open Segment", command=self._open_segment).pack(
            anchor="w", pady=8
        )
        Label(controls, textvariable=self.segment_status_var, fg="blue").pack(
            anchor="w", pady=(0, 8)
        )

        Label(self.window, textvariable=self.status_var, fg="green").pack(
            side=BOTTOM, fill=X, padx=8, pady=4
        )

        history_frame = Frame(self.window)
        history_frame.pack(side=BOTTOM, fill=BOTH, expand=True, padx=8, pady=4)
        Label(history_frame, text="Recent history:").pack(anchor="w")
        history_scroll = Scrollbar(history_frame)
        history_scroll.pack(side=RIGHT, fill=Y)
        self.history_text = Text(
            history_frame, height=8, yscrollcommand=history_scroll.set, state="disabled"
        )
        self.history_text.pack(side=LEFT, fill=BOTH, expand=True)
        history_scroll.config(command=self.history_text.yview)

    def _populate_files(self) -> None:
        """Populate the listbox with available FIF files."""

        try:
            self.fif_files = find_fif_files(self.root_path, provided=self.provided_files)
        except FileNotFoundError as exc:
            messagebox.showerror("No FIF files", str(exc))
            self.window.destroy()
            return

        for path in self.fif_files:
            self.file_list.insert(END, path.name)

        if len(self.fif_files) == 1:
            self.file_list.selection_set(0)
            self._on_file_select()

    def _refresh_history(self) -> None:
        """Load history log entries into the UI widget."""

        entries = read_history(self.log_path)
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", END)
        for entry in entries:
            self.history_text.insert(END, f"{entry}\n")
        self.history_text.configure(state="disabled")
        if entries:
            self.history_text.see(END)

    def _on_file_select(self, event: object | None = None) -> None:  # noqa: ARG002
        """Handle a selection change in the listbox."""

        selection = self.file_list.curselection()
        if not selection:
            return
        idx = selection[0]
        self.selected_file = self.fif_files[idx]
        csv_path = self.selected_file.with_suffix(".csv")
        self.annotation_frame = load_annotation_frame(csv_path)

        raw = mne.io.read_raw_fif(self.selected_file, preload=False)
        self.total_duration = float(raw.times[-1])
        count = len(self.annotation_frame)

        self.info_var.set(
            f"Selected: {self.selected_file.name}\n"
            f"Duration: {self.total_duration:.1f} s\n"
            f"Annotations: {count}\n"
            f"Last edit: {last_edit_timestamp(csv_path)}"
        )
        self.segment_status_var.set("")
        self.status_var.set("Enter a segment and click Open Segment.")
        self._log_history(f"Selected file {self.selected_file.name} (annotations: {count})")

    def _log_history(self, message: str) -> None:
        """Record a history message to the log file and refresh the panel."""

        logger.info(message)
        self._refresh_history()

    def _validate_time_window(self) -> Optional[tuple[float, float]]:
        """Return a validated time window or ``None`` if invalid."""

        if self.total_duration is None:
            messagebox.showwarning("No file", "Please select a FIF file first.")
            return None

        start_raw = self.start_entry.get().strip()
        end_raw = self.end_entry.get().strip()
        require_segment = len(self.annotation_frame) > self.annotation_threshold

        if not start_raw and not end_raw:
            if require_segment:
                messagebox.showwarning(
                    "Segment required",
                    "This file has many annotations; please enter start and end times.",
                )
                return None
            return 0.0, float(self.total_duration)

        try:
            start = float(start_raw)
            end = float(end_raw)
        except ValueError:
            messagebox.showerror("Invalid input", "Start and End must be numbers.")
            return None

        if start < 0 or end <= start or (self.total_duration and end > self.total_duration):
            messagebox.showerror(
                "Invalid range",
                "Ensure 0 <= start < end and the end does not exceed the recording length.",
            )
            return None

        return start, end

    def _open_segment(self) -> None:
        """Launch the MNE browser for the selected segment."""

        if self.selected_file is None:
            messagebox.showwarning("No file", "Please select a FIF file first.")
            return

        window = self._validate_time_window()
        if window is None:
            return
        start, end = window
        annotated_before = segment_already_annotated(self.annotation_frame, start, end)
        status = "already annotated" if annotated_before else "new segment"
        self.segment_status_var.set(
            f"Segment {start:.1f}–{end:.1f} s status: {status}"
        )

        raw = mne.io.read_raw_fif(self.selected_file, preload=True)
        raw.set_annotations(annotations_from_frame(self.annotation_frame))
        session = AnnotationSession(
            fif_path=self.selected_file,
            csv_path=self.selected_file.with_suffix(".csv"),
            start=start,
            end=end,
            annotated_before=annotated_before,
        )

        segment_frame = launch_browser_and_collect(raw, session)

        if not messagebox.askyesno(
            "Save annotations",
            "Merge and save these annotations to the CSV?",
        ):
            self.status_var.set("Changes discarded at user request.")
            return

        merged, dropped = merge_annotations(self.annotation_frame, segment_frame, start, end)
        save_annotations(session.csv_path, merged)
        added = len(segment_frame)
        self.annotation_frame = merged
        self.status_var.set(f"Saved annotations to {session.csv_path}")
        self.info_var.set(
            f"Selected: {self.selected_file.name}\n"
            f"Duration: {self.total_duration:.1f} s\n"
            f"Annotations: {len(self.annotation_frame)}\n"
            f"Last edit: {last_edit_timestamp(session.csv_path)}"
        )
        self._log_history(
            (
                f"Saved segment {start:.1f}-{end:.1f} s for {self.selected_file.name} "
                f"(added: {added}, dropped: {dropped}, total: {len(self.annotation_frame)})"
            )
        )

    def run(self) -> None:
        """Start the Tkinter main loop."""

        self.window.mainloop()


def run_ui(root: str, annotation_threshold: int, files: Iterable[str] | None, start: float | None, end: float | None) -> None:
    """Entry point retained for CLI parity; not used by the Tkinter GUI."""

    del root, annotation_threshold, files, start, end
    raise RuntimeError("The CLI workflow has been replaced by the Tkinter GUI.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""

    parser = argparse.ArgumentParser(description="Annotate FIF files in segments.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
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
    app = AnnotationApp(root_path=Path(args.root), provided_files=args.files, annotation_threshold=args.threshold)
    app.run()


if __name__ == "__main__":
    main()
