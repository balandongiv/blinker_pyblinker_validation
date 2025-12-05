"""Tkinter UI for Raja dataset annotations mirroring the Murat flow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

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
from .constants import DEFAULT_CVAT_ROOT, DEFAULT_DATA_ROOT, DEFAULT_SAMPLING_RATE
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

        self.start_entry: Entry
        self.end_entry: Entry
        self.subject_list: Listbox
        self.session_list: Listbox
        self.history_text: Text

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

        button_frame = Frame(control_frame)
        button_frame.pack(fill=X, pady=5)
        Button(button_frame, text="Open Browser", command=self._open_browser).pack(side=LEFT, padx=5)
        Button(button_frame, text="Refresh", command=self._refresh_current_session).pack(side=LEFT, padx=5)

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

    def _open_browser(self) -> None:
        if self.selected_session is None:
            messagebox.showerror("No session", "Please select a session first.")
            return
        start, end = self._current_range()

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
        ann_inside, inside, outside, _ = prepare_annotations_for_window(frame, start, end)

        ann_manual = launch_browser_and_collect(
            raw,
            ann_inside,
            session_cfg,
            start,
            picks=None,
            duration=end - start,
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
