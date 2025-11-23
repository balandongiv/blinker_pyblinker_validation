r"""Tkinter GUI for segment-wise MNE annotation management.

Flowchart-style walkthrough of the GUI logic:

1. Launch and discover FIF files
   → Prefer the configured dataset roots listed in ``config/dataset_paths.json``
     (defaulting to ``D:\\dataset\\murat_2018`` then
     ``G:\\Other computers\\My Computer\\murat_2018``) and traverse all nested
     folders for ``*.fif`` files. An explicit ``--root`` or ``--files`` list can
     override the default search.
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
   → If both fields are empty, the app defaults to the full recording duration
     regardless of annotation volume.
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
   → The app isolates the annotations in the edited window, lets the user edit
     only that temporary slice, then merges the updated slice back together
     with untouched annotations outside the window.
   → Before overwriting the main CSV, it writes a timestamped backup to a
     ``backups/`` folder beside the CSV.
   → A logger writes major events (segments processed, additions/removals,
     totals) to a history panel and to a file beside the dataset root.
   → The main CSV always reflects the latest global annotation set for the FIF
     file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import json

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
    Entry,
    Frame,
    Label,
    Listbox,
    Scrollbar,
    Button,
    Radiobutton,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
)

from src.utils.annotations import annotations_to_frame, summarize_annotation_changes

from .annotation_io import annotations_from_frame, load_annotation_frame, save_annotations
from .browser import launch_browser_and_collect
from .constants import ANNOTATION_COLUMNS, DATASET_PATHS_FILE, DEFAULT_ROOT_CANDIDATES
from .discovery import find_fif_files
from .logging_utils import (
    configure_logger,
    last_edit_timestamp,
    latest_file_history,
    latest_remark,
    logger,
    read_history,
)
from .session import AnnotationSession
from .segment_utils import (
    merge_updated_annotations,
    prepare_annotations_for_window,
    segment_already_annotated,
)

def default_root_candidates() -> list[Path]:
    """Return configured dataset roots from JSON or fall back to defaults."""

    if DATASET_PATHS_FILE.exists():
        try:
            content = json.loads(DATASET_PATHS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse %s; using built-in dataset roots.", DATASET_PATHS_FILE
            )
        else:
            if isinstance(content, dict) and isinstance(content.get("paths"), list):
                candidates = [
                    Path(entry).expanduser()
                    for entry in content["paths"]
                    if isinstance(entry, str) and entry.strip()
                ]
                if candidates:
                    return candidates

    return list(DEFAULT_ROOT_CANDIDATES)


class AnnotationApp:
    """Tkinter GUI controller for annotation management."""

    def __init__(
        self,
        root_path: Path | None = None,
        provided_files: Optional[Iterable[str]] = None,
        annotation_threshold: int = 100,
    ) -> None:
        self.provided_files = provided_files
        self.annotation_threshold = annotation_threshold

        candidate_roots = [root_path] if root_path else default_root_candidates()

        self.window = Tk()
        self.window.title("MNE Annotation Helper")

        resolved_root = self._resolve_root_path(candidate_roots)
        self.status_var = StringVar(value="Select a FIF file to begin.")
        self.info_var = StringVar(value="No file selected.")
        self.segment_status_var = StringVar(value="")
        self.root_var = StringVar(value=str(resolved_root))

        self.file_search_var = StringVar(value="")
        self.file_search_var.trace_add("write", lambda *_: self._filter_files())

        self.plot_channel_mode = StringVar(value="all")

        self._set_root_path(resolved_root)
        self.latest_remark: str | None = latest_remark(self.log_path)

        self.start_entry: Entry
        self.end_entry: Entry
        self.file_list: Listbox
        self.history_text: Text
        self.annotation_filter: Listbox
        self.remark_entry: Entry
        self.channel_entry: Entry
        self.filtered_files: list[Path] = []

        self.annotation_frame = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        self.total_duration: float | None = None
        self.selected_file: Path | None = None

        self._build_ui()
        self._populate_files()
        self._refresh_history()

    def _set_root_path(self, new_root: Path) -> None:
        """Update the active dataset root and log destination."""

        self.root_path = new_root
        self.root_var.set(str(self.root_path))
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        self.log_path = self.root_path / "logs" / "annotation_ui.log"
        configure_logger(self.log_path)

    def _resolve_root_path(self, candidates: Sequence[Path]) -> Path:
        """Return an existing dataset root, prompting the user if necessary."""

        expanded = [path.expanduser() for path in candidates]
        for path in expanded:
            if path.exists():
                return path

        missing_paths = "\n".join(f"• {path}" for path in expanded)
        messagebox.showwarning(
            "Dataset root not found",
            (
                "None of the configured dataset roots are accessible:\n"
                f"{missing_paths}\n\n"
                "Please select the folder that contains your FIF files."
            ),
            parent=self.window,
        )

        selection = filedialog.askdirectory(parent=self.window, title="Select dataset root")
        if selection:
            return Path(selection).expanduser()

        messagebox.showerror(
            "Dataset root not found",
            (
                "No dataset folder was selected; the application will use the "
                "current working directory instead."
            ),
            parent=self.window,
        )
        return Path.cwd()

    def _build_ui(self) -> None:
        """Construct the Tkinter layout."""

        top_frame = Frame(self.window)
        top_frame.pack(side=TOP, fill=BOTH, expand=True, padx=8, pady=8)

        list_frame = Frame(top_frame)
        list_frame.pack(side=LEFT, fill=BOTH, expand=True)

        root_frame = Frame(list_frame)
        root_frame.pack(fill=X, pady=(0, 6))
        Label(root_frame, text="Dataset root:").pack(side=LEFT)
        root_entry = Entry(root_frame, textvariable=self.root_var, width=50)
        root_entry.pack(side=LEFT, padx=4, fill=X, expand=True)
        Button(root_frame, text="Browse", command=self._choose_root).pack(side=LEFT, padx=(0, 4))
        Button(root_frame, text="Rescan", command=self._populate_files).pack(side=LEFT)

        Label(list_frame, text="Available FIF files:").pack(anchor="w")
        search_frame = Frame(list_frame)
        search_frame.pack(fill=X, pady=(0, 4))
        Label(search_frame, text="Search:").pack(side=LEFT)
        search_entry = Entry(search_frame, textvariable=self.file_search_var)
        search_entry.pack(side=LEFT, fill=X, expand=True, padx=(4, 0))
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

        channel_frame = Frame(controls)
        channel_frame.pack(fill=X, pady=(4, 0))
        Label(channel_frame, text="Channel plotting:").pack(anchor="w")
        channel_radio_frame = Frame(channel_frame)
        channel_radio_frame.pack(anchor="w", fill=X)
        Radiobutton(
            channel_radio_frame,
            text="Plot all channels",
            variable=self.plot_channel_mode,
            value="all",
            command=self._update_channel_entry_state,
        ).pack(anchor="w")
        Radiobutton(
            channel_radio_frame,
            text="Plot selected channels only (comma-separated):",
            variable=self.plot_channel_mode,
            value="selected",
            command=self._update_channel_entry_state,
        ).pack(anchor="w")
        self.channel_entry = Entry(channel_frame, state="disabled")
        self.channel_entry.pack(anchor="w", fill=X, padx=(18, 0), pady=(2, 0))

        Button(controls, text="Open Segment", command=self._open_segment).pack(
            anchor="w", pady=8
        )
        Label(controls, text="Hide annotation labels in plot:").pack(
            anchor="w", pady=(4, 0)
        )
        filter_frame = Frame(controls)
        filter_frame.pack(fill=BOTH, pady=(2, 8))
        filter_scroll = Scrollbar(filter_frame)
        filter_scroll.pack(side=RIGHT, fill=Y)
        self.annotation_filter = Listbox(
            filter_frame,
            selectmode="multiple",
            exportselection=False,
            yscrollcommand=filter_scroll.set,
            height=6,
        )
        self.annotation_filter.pack(side=LEFT, fill=BOTH, expand=True)
        filter_scroll.config(command=self.annotation_filter.yview)
        Label(controls, textvariable=self.segment_status_var, fg="blue").pack(
            anchor="w", pady=(0, 8)
        )

        Label(self.window, textvariable=self.status_var, fg="green").pack(
            side=BOTTOM, fill=X, padx=8, pady=4
        )

        remark_frame = Frame(self.window)
        remark_frame.pack(side=BOTTOM, fill=X, padx=8, pady=(0, 4))
        Label(remark_frame, text="Remark:").pack(side=LEFT)
        self.remark_entry = Entry(remark_frame)
        self.remark_entry.pack(side=LEFT, padx=4, fill=X, expand=True)
        Button(remark_frame, text="Log remark", command=self._submit_remark).pack(
            side=LEFT
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

    def _update_channel_entry_state(self) -> None:
        """Enable or disable channel entry based on the selected plot mode."""

        state = "normal" if self.plot_channel_mode.get() == "selected" else "disabled"
        self.channel_entry.configure(state=state)
        if state == "disabled":
            self.channel_entry.delete(0, END)

    def _populate_files(self) -> None:
        """Populate the listbox with available FIF files."""

        self.root_path = Path(self.root_var.get()).expanduser()
        self.selected_file = None
        self.total_duration = None
        try:
            self.fif_files = find_fif_files(self.root_path, provided=self.provided_files)
        except FileNotFoundError as exc:
            retry = messagebox.askyesno(
                "No FIF files",
                f"{exc}\n\nWould you like to choose a different folder?",
                parent=self.window,
            )
            if retry:
                selection = filedialog.askdirectory(
                    parent=self.window, title="Select dataset root"
                )
                if selection:
                    self._set_root_path(Path(selection).expanduser())
                    self._populate_files()
            return

        self.status_var.set(
            f"Found {len(self.fif_files)} FIF files under {self.root_path}"
        )
        self._log_history(
            f"Scanning root {self.root_path} yielded {len(self.fif_files)} FIF files"
        )
        self._filter_files()

    def _refresh_history(self) -> None:
        """Load history log entries into the UI widget."""

        entries = read_history(self.log_path)
        self.latest_remark = latest_remark(self.log_path)
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", END)
        for entry in entries:
            self.history_text.insert(END, f"{entry}\n")
        self.history_text.configure(state="disabled")
        if entries:
            self.history_text.see(END)

    def _refresh_info_panel(self) -> None:
        """Update the file metadata panel."""

        if self.selected_file is None or self.total_duration is None:
            self.info_var.set("No file selected.")
            return

        csv_path = self.selected_file.with_suffix(".csv")
        count = len(self.annotation_frame)
        recent_edits = latest_file_history(self.log_path, self.selected_file.name)
        history_text = "\n".join(recent_edits) if recent_edits else "None"
        remark_text = self.latest_remark or "None"

        self.info_var.set(
            f"Selected: {self.selected_file.name}\n"
            f"Duration: {self.total_duration:.1f} s\n"
            f"Annotations: {count}\n"
            f"Last edit: {last_edit_timestamp(csv_path)}\n"
            f"Remark: {remark_text}\n"
            f"Recent actions:\n{history_text}"
        )

    def _on_file_select(self, event: object | None = None) -> None:  # noqa: ARG002
        """Handle a selection change in the listbox."""

        selection = self.file_list.curselection()
        if not selection:
            return
        idx = selection[0]
        try:
            self.selected_file = self.filtered_files[idx]
        except IndexError:
            return
        csv_path = self.selected_file.with_suffix(".csv")
        self.annotation_frame = load_annotation_frame(csv_path)
        self._refresh_annotation_filters()

        raw = mne.io.read_raw_fif(self.selected_file, preload=False)
        self.total_duration = float(raw.times[-1])

        self._refresh_info_panel()
        self.segment_status_var.set("")
        self.status_var.set("Enter a segment and click Open Segment.")
        self._log_history(
            f"Selected file {self.selected_file.name} (annotations: {len(self.annotation_frame)})"
        )

    def _log_history(self, message: str) -> None:
        """Record a history message to the log file and refresh the panel."""

        logger.info(message)
        if message.startswith("Remark:"):
            self.latest_remark = message
        self._refresh_history()
        self._refresh_info_panel()

    def _refresh_file_list(self, previous_selection: Path | None = None) -> None:
        """Refresh the visible list of FIF files based on the current filter."""

        self.file_list.delete(0, END)
        for path in self.filtered_files:
            self.file_list.insert(END, path.name)

        if previous_selection and previous_selection in self.filtered_files:
            idx = self.filtered_files.index(previous_selection)
            self.file_list.selection_set(idx)
            self._on_file_select()
        elif len(self.filtered_files) == 1:
            self.file_list.selection_set(0)
            self._on_file_select()

    def _filter_files(self, *_: object) -> None:
        """Filter the displayed FIF list based on the search query."""

        query = self.file_search_var.get().strip().lower()
        previous_selection = self.selected_file
        self.filtered_files = [
            path for path in getattr(self, "fif_files", []) if query in path.name.lower()
        ]
        self._refresh_file_list(previous_selection=previous_selection)

    def _submit_remark(self) -> None:
        """Persist a user-provided remark to the activity log."""

        remark = self.remark_entry.get().strip()
        if not remark:
            messagebox.showinfo("Empty remark", "Please type a remark before logging.")
            return

        self._log_history(f"Remark: {remark}")
        self.remark_entry.delete(0, END)

    def _choose_root(self) -> None:
        """Open a folder chooser to set the dataset root before rescanning."""

        selection = filedialog.askdirectory(initialdir=str(self.root_path))
        if not selection:
            return

        self.root_var.set(selection)
        self._populate_files()

    def _validate_time_window(self) -> Optional[tuple[float, float]]:
        """Return a validated time window or ``None`` if invalid."""

        if self.total_duration is None:
            messagebox.showwarning("No file", "Please select a FIF file first.")
            return None

        start_raw = self.start_entry.get().strip()
        end_raw = self.end_entry.get().strip()
        if not start_raw and not end_raw:
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

    def _refresh_annotation_filters(self) -> None:
        """Populate the annotation filter listbox with available labels."""

        labels: list[str] = []
        if not self.annotation_frame.empty:
            labels = sorted(
                {
                    desc
                    for desc in self.annotation_frame["description"].dropna().astype(str)
                    if desc
                }
            )

        self.annotation_filter.delete(0, END)
        for label in labels:
            self.annotation_filter.insert(END, label)

        self.annotation_filter.configure(state="normal" if labels else "disabled")

    def _selected_labels_to_skip(self) -> Set[str]:
        """Return the set of labels selected for exclusion from plotting."""

        if not self.annotation_filter.size():
            return set()

        return {
            self.annotation_filter.get(idx)
            for idx in self.annotation_filter.curselection()
        }

    def _determine_channel_picks(self, raw: mne.io.Raw) -> list[str] | None:
        """Return channel picks or an empty list to signal invalid input."""

        if self.plot_channel_mode.get() != "selected":
            return None

        requested = [
            channel.strip()
            for channel in self.channel_entry.get().split(",")
            if channel.strip()
        ]
        if not requested:
            messagebox.showwarning(
                "Channel selection",
                "Enter one or more channel names or switch to plotting all channels.",
            )
            return []

        missing = [channel for channel in requested if channel not in raw.ch_names]
        if missing:
            available_preview = ", ".join(raw.ch_names[:10])
            if len(raw.ch_names) > 10:
                available_preview += ", ..."
            messagebox.showwarning(
                "Channel selection",
                (
                    "The following channels were not found in this recording: "
                    f"{', '.join(missing)}.\n"
                    "Please choose channels from the available list "
                    f"(e.g., {available_preview})."
                ),
            )
            return []

        return requested

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

        annotations = mne.Annotations(
            onset=self.annotation_frame["onset"].astype(float).to_numpy(),
            duration=self.annotation_frame["duration"].fillna(0).astype(float).to_numpy(),
            description=self.annotation_frame["description"].fillna("").astype(str).tolist(),
        )

        session = AnnotationSession(
            fif_path=self.selected_file,
            csv_path=self.selected_file.with_suffix(".csv"),
            start=start,
            end=end,
            annotated_before=annotated_before,
        )

        raw = mne.io.read_raw_fif(self.selected_file, preload=True)
        raw.set_annotations(annotations)
        frame = annotations_to_frame(raw.annotations)
        skip_labels = self._selected_labels_to_skip()
        ann_inside, inside, outside, _ = prepare_annotations_for_window(
            frame, start, end, skip_labels
        )

        picks = self._determine_channel_picks(raw)
        if picks == []:
            return

        ann_manual = launch_browser_and_collect(raw, ann_inside, session, start, picks)

        updated_inside = annotations_to_frame(ann_manual)
        change_summary = summarize_annotation_changes(inside, updated_inside)
        has_changes = any(
            [
                change_summary.added,
                change_summary.removed,
                change_summary.changed,
            ]
        )

        if not has_changes:
            self.status_var.set("No Addition and Changes has been Made")
            self._log_history("No Addition and Changes has been Made")
            return

        merged = merge_updated_annotations(updated_inside, outside)
        annot_merged = annotations_from_frame(merged)
        raw.set_annotations(annot_merged)
        if not messagebox.askyesno(
            "Save annotations",
            (
                "Merge and save these annotations to the CSV?\n"
                "Yes: Save the latest changes to the CSV.\n"
                "No: Keep the existing CSV unchanged for now (you will be asked to confirm discarding)."
            ),
        ):
            discard = messagebox.askyesno(
                "Discard changes?",
                (
                    "You have made changes to these annotations. Discard them without saving?\n"
                    "Yes: No changes will be made to the CSV.\n"
                    "No: The latest changes will be saved to the CSV."
                ),
            )
            if discard:
                self.status_var.set("Changes discarded at user request.")
                return

        save_annotations(session.csv_path, merged)

        self.annotation_frame = merged
        self._refresh_annotation_filters()
        self.status_var.set(
            (
                f"Saved annotations to {session.csv_path} "
                f"(added: {change_summary.added}, removed: {change_summary.removed}, "
                f"changed: {change_summary.changed})"
            )
        )
        self._log_history(
            (
                f"Saved segment {start:.1f}-{end:.1f} s for {self.selected_file.name} "
                f"(added: {change_summary.added}, removed: {change_summary.removed}, changed: {change_summary.changed}, "
                f"total: {len(self.annotation_frame)})"
            )
        )

    def run(self) -> None:
        """Start the Tkinter main loop."""

        self.window.mainloop()


def run_ui(
    root: str,
    annotation_threshold: int,
    files: Iterable[str] | None,
    start: float | None,
    end: float | None,
) -> None:
    """Entry point retained for CLI parity; not used by the Tkinter GUI."""

    del root, annotation_threshold, files, start, end
    raise RuntimeError("The CLI workflow has been replaced by the Tkinter GUI.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""

    parser = argparse.ArgumentParser(description="Annotate FIF files in segments.")
    parser.add_argument(
        "--root",
        default=str(default_root_candidates()[0]),
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
