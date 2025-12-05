"""Tkinter UI for Raja dataset annotations mirroring the Murat flow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Set

import mne
import pandas as pd
from tkinter import (
    BOTH,
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
    Radiobutton,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    messagebox,
)

from src.utils.annotations import annotations_to_frame, summarize_annotation_changes

from src.ui_murat.annotation_io import (
    annotations_from_frame,
    load_annotation_frame,
    save_annotations,
)
from src.ui_murat.browser import launch_browser_and_collect
from src.ui_murat.session import AnnotationSession
from src.ui_murat.segment_utils import (
    merge_updated_annotations,
    prepare_annotations_for_window,
    segment_already_annotated,
)

from .annotation_import import AnnotationImportError, ensure_annotations
from .constants import (
    DEFAULT_CHANNEL_PICKS,
    DEFAULT_CVAT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_SAMPLING_RATE,
)
from .discovery import RajaDataset, SessionInfo

logger = logging.getLogger(__name__)


class RajaAnnotationApp:
    """Tkinter GUI controller for Raja annotations."""

    def __init__(self, data_root: Path, cvat_root: Path, *, sampling_rate: float) -> None:
        self.data_root = data_root
        self.cvat_root = cvat_root
        self.sampling_rate = sampling_rate
        self.dataset = RajaDataset(self.data_root)

        self.window = Tk()
        self.window.title("Raja Annotation Helper")

        self.status_var = StringVar(value="Select a session to begin.")
        self.info_var = StringVar(value="No session selected.")
        self.segment_status_var = StringVar(value="")
        self.plot_channel_mode = StringVar(value="selected")
        self.plot_duration_var = StringVar(value="")

        self.start_entry: Entry
        self.end_entry: Entry
        self.subject_list: Listbox
        self.session_list: Listbox
        self.history_text: Text
        self.annotation_filter: Listbox
        self.channel_entry: Entry
        self.duration_entry: Entry

        self.selected_session: SessionInfo | None = None
        self.annotation_frame = pd.DataFrame()
        self.total_duration: float | None = None

        self._build_ui()
        self._populate_subjects()

    def _build_ui(self) -> None:
        header = Frame(self.window)
        header.pack(side=TOP, fill=X)
        Label(header, textvariable=self.status_var, fg="blue").pack(side=LEFT, padx=5, pady=5)

        main = Frame(self.window)
        main.pack(fill=BOTH, expand=True)

        # Subject list
        subject_frame = Frame(main)
        subject_frame.pack(side=LEFT, fill=Y)
        Label(subject_frame, text="Subjects").pack()
        subject_scroll = Scrollbar(subject_frame)
        subject_scroll.pack(side=RIGHT, fill=Y)
        self.subject_list = Listbox(subject_frame, exportselection=False)
        self.subject_list.pack(side=LEFT, fill=Y)
        self.subject_list.config(yscrollcommand=subject_scroll.set)
        subject_scroll.config(command=self.subject_list.yview)
        self.subject_list.bind("<<ListboxSelect>>", lambda *_: self._handle_subject_selection())

        # Session list
        session_frame = Frame(main)
        session_frame.pack(side=LEFT, fill=Y)
        Label(session_frame, text="Sessions").pack()
        session_scroll = Scrollbar(session_frame)
        session_scroll.pack(side=RIGHT, fill=Y)
        self.session_list = Listbox(session_frame, width=50, exportselection=False)
        self.session_list.pack(side=LEFT, fill=Y)
        self.session_list.config(yscrollcommand=session_scroll.set)
        session_scroll.config(command=self.session_list.yview)
        self.session_list.bind("<<ListboxSelect>>", lambda *_: self._handle_session_selection())

        # Controls
        control_frame = Frame(main)
        control_frame.pack(side=LEFT, fill=BOTH, expand=True)

        info_frame = Frame(control_frame)
        info_frame.pack(fill=X, pady=5)
        Label(info_frame, textvariable=self.info_var, fg="green", wraplength=400, justify="left").pack(
            anchor="w"
        )
        Label(info_frame, textvariable=self.segment_status_var, fg="purple").pack(anchor="w")

        range_frame = Frame(control_frame)
        range_frame.pack(fill=X, pady=5)
        Label(range_frame, text="Start (s)").pack(side=LEFT)
        self.start_entry = Entry(range_frame, width=8)
        self.start_entry.pack(side=LEFT, padx=5)
        Label(range_frame, text="End (s)").pack(side=LEFT)
        self.end_entry = Entry(range_frame, width=8)
        self.end_entry.pack(side=LEFT, padx=5)

        duration_frame = Frame(control_frame)
        duration_frame.pack(fill=X, pady=(0, 4))
        Label(duration_frame, text="Browser duration (s, optional):").pack(side=LEFT)
        self.duration_entry = Entry(duration_frame, width=10, textvariable=self.plot_duration_var)
        self.duration_entry.pack(side=LEFT, padx=(4, 0))
        Label(duration_frame, text="(controls arrow key jump)").pack(side=LEFT, padx=(6, 0))

        channel_frame = Frame(control_frame)
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
        self.channel_entry = Entry(channel_frame, state="normal")
        self.channel_entry.pack(anchor="w", fill=X, padx=(18, 0), pady=(2, 0))
        self.channel_entry.insert(0, ", ".join(DEFAULT_CHANNEL_PICKS))

        button_frame = Frame(control_frame)
        button_frame.pack(fill=X, pady=5)
        Button(button_frame, text="Open Browser", command=self._open_browser).pack(side=LEFT, padx=5)
        Button(button_frame, text="Refresh", command=self._refresh_current_session).pack(side=LEFT, padx=5)

        Label(control_frame, text="Hide annotation labels in plot:").pack(
            anchor="w", pady=(4, 0)
        )
        filter_frame = Frame(control_frame)
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

        self._update_channel_entry_state()

        history_frame = Frame(control_frame)
        history_frame.pack(fill=BOTH, expand=True)
        Label(history_frame, text="History").pack(anchor="w")
        self.history_text = Text(history_frame, height=8)
        self.history_text.pack(fill=BOTH, expand=True)

    def _log(self, message: str) -> None:
        self.history_text.insert(END, message + "\n")
        self.history_text.see(END)
        logger.info(message)

    def _populate_subjects(self) -> None:
        self.subject_list.delete(0, END)
        for subject_id in self.dataset.subjects():
            self.subject_list.insert(END, subject_id)
        if self.dataset.subjects():
            self.subject_list.selection_set(0)
            self._handle_subject_selection()

    def _handle_subject_selection(self) -> None:
        selection = self.subject_list.curselection()
        if not selection:
            return
        subject_id = self.subject_list.get(selection[0])
        sessions = self.dataset.sessions_for(subject_id)
        self.session_list.delete(0, END)
        for session in sessions:
            self.session_list.insert(END, f"{session.session_name} ({session.fif_path.name})")
        if sessions:
            self.session_list.selection_set(0)
            self._handle_session_selection()

    def _handle_session_selection(self) -> None:
        selection = self.session_list.curselection()
        subject_sel = self.subject_list.curselection()
        if not selection or not subject_sel:
            return
        subject_id = self.subject_list.get(subject_sel[0])
        session_info = self.dataset.sessions_for(subject_id)[selection[0]]
        self.selected_session = session_info
        self.status_var.set(f"Selected {session_info.subject_id} / {session_info.session_name}")
        self._load_session_data()

    def _load_session_data(self) -> None:
        assert self.selected_session is not None
        try:
            self.annotation_frame = ensure_annotations(
                self.selected_session, self.cvat_root, sampling_rate=self.sampling_rate
            )
            self._log(
                f"Loaded annotations for {self.selected_session.subject_id}/{self.selected_session.session_name}"
            )
        except AnnotationImportError as exc:
            self.annotation_frame = pd.DataFrame(columns=["onset", "duration", "description"])
            messagebox.showwarning("Annotations unavailable", str(exc))
            self._log(str(exc))

        csv_frame = load_annotation_frame(self.selected_session.annotation_csv)
        if not csv_frame.empty:
            self.annotation_frame = csv_frame

        self._refresh_annotation_filters()

        raw = mne.io.read_raw_fif(self.selected_session.fif_path, preload=False)
        self.total_duration = raw.times[-1]
        summary = (
            f"FIF: {self.selected_session.fif_path}\n"
            f"Annotations: {len(self.annotation_frame)} rows\n"
            f"Duration: {self.total_duration:.1f} seconds"
        )
        self.info_var.set(summary)
        self.segment_status_var.set("")

    def _current_range(self) -> tuple[float, float]:
        start_text = self.start_entry.get().strip()
        end_text = self.end_entry.get().strip()
        start = float(start_text) if start_text else 0.0
        if self.total_duration is None:
            end = float(end_text) if end_text else start
        else:
            end = float(end_text) if end_text else self.total_duration
        if end <= start:
            end = start + 1.0
        return start, end

    def _validate_time_window(self) -> tuple[float, float] | None:
        """Validate and return a time window if possible."""

        if self.total_duration is None:
            messagebox.showwarning("No file", "Please select a FIF file first.")
            return None

        start_raw = self.start_entry.get().strip()
        end_raw = self.end_entry.get().strip()
        if not start_raw and not end_raw:
            return 0.0, float(self.total_duration)

        if start_raw and not end_raw:
            try:
                start = float(start_raw)
            except ValueError:
                messagebox.showerror(
                    "Invalid input", "Start must be a number when End is left blank."
                )
                return None

            if start < 0 or (self.total_duration and start >= self.total_duration):
                messagebox.showerror(
                    "Invalid range",
                    ("Ensure 0 <= start < total duration when omitting the End time."),
                )
                return None

            return start, float(self.total_duration)

        if not start_raw and end_raw:
            messagebox.showerror(
                "Invalid input",
                "Provide a Start time or leave both fields blank to use the full recording.",
            )
            return None

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

    def _parse_browser_duration(self) -> tuple[bool, float | None]:
        """Return a validated browser duration or ``None`` when left blank."""

        raw_value = self.plot_duration_var.get().strip()
        if not raw_value:
            return True, None

        try:
            duration = float(raw_value)
        except ValueError:
            messagebox.showerror(
                "Invalid duration", "Browser duration must be a valid number."
            )
            return False, None

        if duration <= 0:
            messagebox.showerror(
                "Invalid duration",
                "Browser duration must be greater than zero.",
            )
            return False, None

        return True, duration

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

    def _update_channel_entry_state(self) -> None:
        """Enable or disable channel entry based on the selected plot mode."""

        state = "normal" if self.plot_channel_mode.get() == "selected" else "disabled"
        self.channel_entry.configure(state=state)
        if state == "disabled":
            self.channel_entry.delete(0, END)

    def _open_browser(self) -> None:
        if self.selected_session is None:
            messagebox.showerror("No session", "Please select a session first.")
            return
        window = self._validate_time_window()
        if window is None:
            return
        start, end = window
        duration_valid, browser_duration = self._parse_browser_duration()
        if not duration_valid:
            return

        annotated_before = segment_already_annotated(self.annotation_frame, start, end)
        status = "already annotated" if annotated_before else "new segment"
        self.segment_status_var.set(f"Segment {start:.1f}-{end:.1f} s status: {status}")

        annotations = annotations_from_frame(self.annotation_frame)
        session_cfg = AnnotationSession(
            fif_path=self.selected_session.fif_path,
            csv_path=self.selected_session.annotation_csv,
            start=start,
            end=end,
            annotated_before=annotated_before,
        )

        raw = mne.io.read_raw_fif(self.selected_session.fif_path, preload=True)
        raw.set_annotations(annotations)
        frame = annotations_to_frame(raw.annotations)
        skip_labels = self._selected_labels_to_skip()
        ann_inside, inside, outside, _ = prepare_annotations_for_window(
            frame, start, end, skip_labels
        )

        picks = self._determine_channel_picks(raw)
        if picks == []:
            return

        ann_manual = launch_browser_and_collect(
            raw,
            ann_inside,
            session_cfg,
            start,
            picks=picks,
            duration=browser_duration,
        )

        updated_inside = annotations_to_frame(ann_manual)
        change_summary = summarize_annotation_changes(inside, updated_inside)
        if not any([change_summary.added, change_summary.removed, change_summary.changed]):
            self._log("No annotation changes detected; skipping save.")
            return

        merged = merge_updated_annotations(updated_inside, outside)
        save_annotations(self.selected_session.annotation_csv, merged)
        self.annotation_frame = merged
        self._log(
            f"Saved annotations for {self.selected_session.subject_id}/{self.selected_session.session_name}"
        )

    def _refresh_current_session(self) -> None:
        if self.selected_session is None:
            return
        self._load_session_data()



def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Raja annotation UI")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cvat-root", type=Path, default=DEFAULT_CVAT_ROOT)
    parser.add_argument("--sampling-rate", type=float, default=DEFAULT_SAMPLING_RATE)
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO)

    app = RajaAnnotationApp(args.data_root, args.cvat_root, sampling_rate=args.sampling_rate)
    app.window.mainloop()


if __name__ == "__main__":
    main()
