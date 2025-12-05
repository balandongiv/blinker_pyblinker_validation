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
    OptionMenu,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    messagebox,
)
from tkinter import ttk

from src.utils.annotations import (
    annotations_to_frame,
    label_change_breakdown,
    summarize_annotation_changes,
)

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
    CHANNEL_ALIASES,
    DEFAULT_CHANNEL_PICKS,
    DEFAULT_CVAT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_PATH_PAIR_CONFIG,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_STATUS_STORE,
)
from .discovery import RajaDataset, SessionInfo
from .path_pairs import PathPair, load_path_pairs
from .status_store import StatusStore

logger = logging.getLogger(__name__)


STATUS_OPTIONS = ("Pending", "Ongoing", "Complete")
STATUS_ORDER = {name: index for index, name in enumerate(STATUS_OPTIONS)}


def subject_sort_key(subject_id: str) -> tuple[int, str]:
    """Return a numeric-first sort key for Raja subject identifiers."""

    if subject_id.lower().startswith("s") and subject_id[1:].isdigit():
        return int(subject_id[1:]), subject_id
    return (10**6, subject_id.lower())


def status_sort_key(status_value: str) -> tuple[int, str]:
    """Sort key that respects the defined status progression."""

    order = STATUS_ORDER.get(status_value, len(STATUS_ORDER))
    return order, status_value.lower()


class RajaAnnotationApp:
    """Tkinter GUI controller for Raja annotations."""

    def __init__(
        self,
        data_root: Path,
        cvat_root: Path,
        *,
        sampling_rate: float,
        path_pairs: list[PathPair] | None = None,
        selected_pair: str | None = None,
        status_store_path: Path | None = None,
    ) -> None:
        self.data_root = data_root
        self.cvat_root = cvat_root
        self.sampling_rate = sampling_rate
        self.path_pairs = path_pairs or [PathPair("Provided paths", data_root, cvat_root)]
        if not self.path_pairs:
            self.path_pairs = [PathPair("Provided paths", data_root, cvat_root)]
        self.active_pair_name = selected_pair or self.path_pairs[0].name
        self.dataset: RajaDataset | None = None

        self.status_store = StatusStore(status_store_path or DEFAULT_STATUS_STORE)

        self.window = Tk()
        self.window.title("Raja Annotation Helper")

        self.status_var = StringVar(value="Select a session to begin.")
        self.info_var = StringVar(value="No session selected.")
        self.segment_status_var = StringVar(value="")
        self.plot_channel_mode = StringVar(value="selected")
        self.plot_duration_var = StringVar(value="")
        self.path_pair_var = StringVar(value=self.active_pair_name)
        self.status_selection_var = StringVar(value=STATUS_OPTIONS[0])
        self.status_detail_var = StringVar(value="Select a recording to update its status.")

        self.session_status: dict[Path, str] = {}
        self.status_editor: ttk.Combobox | None = None
        self.status_editor_var: StringVar | None = None
        self._status_sort_column: str | None = None
        self._status_sort_reverse = False

        self.start_entry: Entry
        self.end_entry: Entry
        self.subject_list: Listbox
        self.session_list: Listbox
        self.history_text: Text
        self.annotation_filter: Listbox
        self.channel_entry: Entry
        self.duration_entry: Entry
        self.status_tree: ttk.Treeview

        self.selected_session: SessionInfo | None = None
        self.annotation_frame = pd.DataFrame()
        self.total_duration: float | None = None

        self._build_ui()
        self._activate_pair(self.path_pair_var.get())

    def _build_ui(self) -> None:
        header = Frame(self.window)
        header.pack(side=TOP, fill=X)
        Label(header, textvariable=self.status_var, fg="blue").pack(side=LEFT, padx=5, pady=5)

        if self.path_pairs:
            pair_frame = Frame(header)
            pair_frame.pack(side=RIGHT, padx=5)
            Label(pair_frame, text="Path pair:").pack(side=LEFT)
            OptionMenu(
                pair_frame,
                self.path_pair_var,
                *[pair.name for pair in self.path_pairs],
                command=lambda *_: self._on_pair_change(),
            ).pack(side=LEFT)

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=BOTH, expand=True)

        annotate_tab = Frame(notebook)
        status_tab = Frame(notebook)

        notebook.add(annotate_tab, text="Annotate")
        notebook.add(status_tab, text="FIF Status")

        self._build_annotation_tab(annotate_tab)
        self._build_status_tab(status_tab)

    def _build_annotation_tab(self, parent: Frame) -> None:
        main = Frame(parent)
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

    def _build_status_tab(self, parent: Frame) -> None:
        Label(
            parent,
            text=(
                "Track FIF recording status. Select a session to set its status to "
                "Pending (default), Ongoing, or Complete."
            ),
            wraplength=600,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(8, 4))

        tree_frame = Frame(parent)
        tree_frame.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))

        columns = ("subject", "session", "status")
        self.status_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
        for column, heading in zip(columns, ["Subject", "Session", "Status"]):
            self.status_tree.heading(
                column, text=heading, command=lambda c=column: self._sort_status_table(c)
            )
            self.status_tree.column(column, width=180 if column != "status" else 120, anchor="w")

        scrollbar = Scrollbar(tree_frame, orient="vertical", command=self.status_tree.yview)
        self.status_tree.configure(yscrollcommand=scrollbar.set)
        self.status_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.status_tree.bind("<<TreeviewSelect>>", lambda *_: self._on_status_tree_select())
        self.status_tree.bind("<Button-1>", self._on_status_tree_click)

        control_frame = Frame(parent)
        control_frame.pack(fill=X, padx=8, pady=(0, 8))

        Label(control_frame, textvariable=self.status_detail_var, fg="green", wraplength=400).pack(
            side=LEFT, padx=(0, 10)
        )

        OptionMenu(
            control_frame,
            self.status_selection_var,
            *STATUS_OPTIONS,
        ).pack(side=LEFT, padx=(0, 6))

        Button(control_frame, text="Update Status", command=self._set_selected_status).pack(side=LEFT)

    def _log(self, message: str) -> None:
        self.history_text.insert(END, message + "\n")
        self.history_text.see(END)
        logger.info(message)

    def _initialize_session_statuses(self) -> None:
        """Reset FIF status tracking for the active dataset."""

        existing_status = {path.resolve(): status for path, status in self.session_status.items()}
        stored_status = self.status_store.load_for_root(self.data_root)
        self.session_status = {}
        if self.dataset is None:
            return

        for session in self.dataset.all_sessions():
            resolved_fif = session.fif_path.resolve()
            self.session_status[resolved_fif] = stored_status.get(
                resolved_fif,
                existing_status.get(resolved_fif, STATUS_OPTIONS[0]),
            )

    def _refresh_status_table(self) -> None:
        self._clear_status_table()
        if self.dataset is None:
            return

        for session in self.dataset.all_sessions():
            resolved_fif = session.fif_path.resolve()
            status = self.session_status.get(resolved_fif, STATUS_OPTIONS[0])
            self.status_tree.insert(
                "",
                "end",
                iid=str(resolved_fif),
                values=(session.subject_id, session.session_name, status),
            )
        if self._status_sort_column:
            self._sort_status_table(self._status_sort_column, toggle=False)
        else:
            self._sort_status_table("subject", toggle=False)

    def _clear_status_table(self) -> None:
        self._destroy_status_editor()
        if hasattr(self, "status_tree"):
            for child in self.status_tree.get_children():
                self.status_tree.delete(child)
        self.status_detail_var.set("Select a recording to update its status.")

    def _on_status_tree_click(self, event) -> None:  # type: ignore[override]
        self._destroy_status_editor()
        region = self.status_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.status_tree.identify_column(event.x)
        if column != "#3":
            return
        row_id = self.status_tree.identify_row(event.y)
        if not row_id:
            return
        self.status_tree.selection_set(row_id)
        self._begin_status_edit(row_id)

    def _on_status_tree_select(self) -> None:
        selection = self.status_tree.selection()
        if not selection:
            self.status_selection_var.set(STATUS_OPTIONS[0])
            self.status_detail_var.set("Select a recording to update its status.")
            return

        item_id = selection[0]
        values = self.status_tree.item(item_id, "values")
        if len(values) >= 3:
            self.status_selection_var.set(values[2])
            self.status_detail_var.set(
                f"Selected {values[0]} / {values[1]} (status: {values[2]})."
            )

    def _begin_status_edit(self, item_id: str) -> None:
        bbox = self.status_tree.bbox(item_id, "status")
        if not bbox:
            return
        x, y, width, height = bbox
        self._destroy_status_editor()
        current = self.status_tree.set(item_id, "status")
        self.status_editor_var = StringVar(value=current)
        self.status_editor = ttk.Combobox(
            self.status_tree,
            values=STATUS_OPTIONS,
            textvariable=self.status_editor_var,
            state="readonly",
        )
        self.status_editor.place(x=x, y=y, width=width, height=height)
        self.status_editor.bind(
            "<<ComboboxSelected>>", lambda *_: self._commit_status_edit(item_id)
        )
        self.status_editor.bind("<FocusOut>", lambda *_: self._destroy_status_editor())
        self.status_editor.focus_set()

    def _commit_status_edit(self, item_id: str) -> None:
        if not self.status_editor_var:
            return
        new_status = self.status_editor_var.get()
        values = list(self.status_tree.item(item_id, "values"))
        if len(values) >= 3:
            self._update_status(item_id, values, new_status)
        self._destroy_status_editor()

    def _set_selected_status(self) -> None:
        selection = self.status_tree.selection()
        if not selection:
            messagebox.showinfo(
                "No recording selected", "Please select a recording in the table to update its status."
            )
            return

        new_status = self.status_selection_var.get()
        item_id = selection[0]
        values = list(self.status_tree.item(item_id, "values"))
        if len(values) >= 3:
            self._update_status(item_id, values, new_status)

    def _update_status(self, item_id: str, values: list[str], new_status: str) -> None:
        values[2] = new_status
        self.status_tree.item(item_id, values=values)
        fif_path = Path(item_id).resolve()
        self.session_status[fif_path] = new_status
        self.status_store.update_status(self.data_root, fif_path, new_status)
        self.status_selection_var.set(new_status)
        if (
            self.selected_session
            and self.selected_session.fif_path.resolve() == fif_path
        ):
            self._update_info_status_line(new_status)
        self.status_detail_var.set(
            f"Status for {values[0]} / {values[1]} set to {new_status}."
        )

    def _destroy_status_editor(self) -> None:
        if self.status_editor is not None:
            self.status_editor.destroy()
            self.status_editor = None
            self.status_editor_var = None

    def _sort_status_table(self, column: str, *, toggle: bool = True) -> None:
        reverse = self._status_sort_reverse if self._status_sort_column == column else False
        if toggle:
            reverse = not reverse if self._status_sort_column == column else False
        self._status_sort_column = column
        self._status_sort_reverse = reverse

        children = [
            (self.status_tree.item(child, "values"), child) for child in self.status_tree.get_children("")
        ]

        def key_fn(item: tuple[tuple[str, str, str], str]) -> tuple[int, str] | str:
            values, _ = item
            if column == "subject":
                return subject_sort_key(values[0])
            if column == "status":
                return status_sort_key(values[2])
            return values[1].lower()

        for index, (_, iid) in enumerate(sorted(children, key=key_fn, reverse=reverse)):
            self.status_tree.move(iid, "", index)

    def _sync_status_selection(self, fif_path: Path) -> None:
        if not hasattr(self, "status_tree"):
            return

        item_id = str(fif_path.resolve())
        if self.status_tree.exists(item_id):
            self.status_tree.selection_set(item_id)
            self.status_tree.see(item_id)
            self._on_status_tree_select()

    def _update_info_status_line(self, status_value: str) -> None:
        if not self.info_var.get():
            return
        lines = self.info_var.get().splitlines()
        if lines and lines[-1].startswith("Status:"):
            lines[-1] = f"Status: {status_value}"
        else:
            lines.append(f"Status: {status_value}")
        self.info_var.set("\n".join(lines))

    def _pair_by_name(self, name: str) -> PathPair | None:
        for pair in self.path_pairs:
            if pair.name == name:
                return pair
        return None

    def _clear_lists(self) -> None:
        self.subject_list.delete(0, END)
        self.session_list.delete(0, END)
        self.info_var.set("Dataset unavailable for the selected path pair.")
        self.selected_session = None
        self._clear_status_table()

    def _on_pair_change(self) -> None:
        self._activate_pair(self.path_pair_var.get())

    def _activate_pair(self, pair_name: str) -> None:
        pair = self._pair_by_name(pair_name) or self.path_pairs[0]
        self.path_pair_var.set(pair.name)
        self.data_root = pair.data_root
        self.cvat_root = pair.cvat_root
        self.active_pair_name = pair.name
        self.status_var.set(
            f"Using path pair '{pair.name}' (data: {self.data_root}, CVAT: {self.cvat_root})"
        )
        try:
            self.dataset = RajaDataset(self.data_root)
        except FileNotFoundError as exc:
            messagebox.showerror("Dataset unavailable", str(exc))
            self.dataset = None
            self._clear_lists()
            return

        self._initialize_session_statuses()
        self._populate_subjects()
        self._refresh_status_table()

    def _populate_subjects(self) -> None:
        if self.dataset is None:
            return
        self.subject_list.delete(0, END)
        for subject_id in self.dataset.subjects():
            self.subject_list.insert(END, subject_id)
        if self.dataset.subjects():
            self.subject_list.selection_set(0)
            self._handle_subject_selection()

    def _handle_subject_selection(self) -> None:
        if self.dataset is None:
            return
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
        if self.dataset is None:
            return
        selection = self.session_list.curselection()
        subject_sel = self.subject_list.curselection()
        if not selection or not subject_sel:
            return
        subject_id = self.subject_list.get(subject_sel[0])
        session_info = self.dataset.sessions_for(subject_id)[selection[0]]
        self.selected_session = session_info
        self.status_var.set(f"Selected {session_info.subject_id} / {session_info.session_name}")
        self._load_session_data()
        self._sync_status_selection(session_info.fif_path)

    def _load_session_data(self) -> None:
        if self.selected_session is None or self.dataset is None:
            messagebox.showwarning("No session", "Please select a valid session.")
            return
        # Reset UI fields for the newly selected session so ranges default to the
        # full recording unless the user specifies otherwise.
        self.start_entry.delete(0, END)
        self.end_entry.delete(0, END)
        annotation_source = "Annotations unavailable"
        try:
            self.annotation_frame, annotation_source = ensure_annotations(
                self.selected_session, self.cvat_root, sampling_rate=self.sampling_rate
            )
            self._log(
                f"Loaded annotations for {self.selected_session.subject_id}/{self.selected_session.session_name}"
            )
            self._log(f"Annotation source: {annotation_source}")
        except AnnotationImportError as exc:
            self.annotation_frame = pd.DataFrame(columns=["onset", "duration", "description"])
            messagebox.showwarning("Annotations unavailable", str(exc))
            annotation_source = f"Annotations unavailable ({exc})"
            self._log(annotation_source)

        csv_frame = load_annotation_frame(self.selected_session.annotation_csv)
        if not csv_frame.empty:
            self.annotation_frame = csv_frame

        self._refresh_annotation_filters()

        raw = mne.io.read_raw_fif(self.selected_session.fif_path, preload=False)
        self.total_duration = raw.times[-1]
        status_value = self.session_status.get(
            self.selected_session.fif_path.resolve(), "Pending"
        )
        summary = (
            f"FIF: {self.selected_session.fif_path}\n"
            f"Annotations: {len(self.annotation_frame)} rows\n"
            f"Annotation source: {annotation_source}\n"
            f"Duration: {self.total_duration:.1f} seconds\n"
            f"Status: {status_value}"
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
        try:
            start = float(start_raw) if start_raw else 0.0
        except ValueError:
            messagebox.showerror("Invalid input", "Start must be a number if provided.")
            return None

        try:
            end = float(end_raw) if end_raw else float(self.total_duration)
        except ValueError:
            messagebox.showerror("Invalid input", "End must be a number if provided.")
            return None

        if start < 0 or (self.total_duration and start >= self.total_duration):
            messagebox.showerror(
                "Invalid range",
                "Ensure the start time is within the recording (0 <= start < total duration).",
            )
            return None

        if end <= start:
            messagebox.showerror(
                "Invalid range",
                "Ensure end is greater than start or leave End blank to use the full recording.",
            )
            return None

        if self.total_duration and end > self.total_duration:
            messagebox.showerror(
                "Invalid range",
                "End time cannot exceed the recording length; leave it blank to use the full recording.",
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

        lower_raw = {name.lower(): name for name in raw.ch_names}
        aliases = {alias.lower(): target for alias, target in CHANNEL_ALIASES.items()}

        resolved: list[str] = []
        missing: list[str] = []

        for channel in requested:
            if channel in raw.ch_names:
                resolved.append(channel)
                continue

            lower = channel.lower()
            if lower in aliases and aliases[lower] in raw.ch_names:
                resolved.append(aliases[lower])
                continue

            if lower in lower_raw:
                resolved.append(lower_raw[lower])
                continue

            missing.append(channel)
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

        return resolved

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

        self._log(
            f"Launching browser with {len(ann_inside)} annotations in view "
            f"({len(frame)} total loaded)."
        )
        self.segment_status_var.set(
            f"Segment {start:.1f}-{end:.1f} s status: {status}; "
            f"annotations in view: {len(ann_inside)}"
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
        label_breakdown = label_change_breakdown(inside, updated_inside)
        has_changes = any(
            [change_summary.added, change_summary.removed, change_summary.changed]
        )
        if not has_changes:
            message = "No annotation changes detected; skipping save."
            self._log(message)
            self.status_var.set(message)
            return

        merged = merge_updated_annotations(updated_inside, outside)
        save_annotations(self.selected_session.annotation_csv, merged)
        self.annotation_frame = merged
        summary_msg = (
            f"Saved segment {start:.1f}-{end:.1f} s for "
            f"{self.selected_session.subject_id}/{self.selected_session.session_name} "
            f"(added: {change_summary.added}, removed: {change_summary.removed}, "
            f"changed: {change_summary.changed}, total: {len(self.annotation_frame)})"
        )
        self._log(summary_msg)
        detail_parts = []
        if label_breakdown.added_counts:
            added_detail = ", ".join(
                f"{label} x{count}" for label, count in label_breakdown.added_counts.items()
            )
            detail_parts.append(f"Added by label: {added_detail}")
        if label_breakdown.removed_counts:
            removed_detail = ", ".join(
                f"{label} x{count}" for label, count in label_breakdown.removed_counts.items()
            )
            detail_parts.append(f"Removed by label: {removed_detail}")
        if label_breakdown.changed_labels:
            changed_detail = ", ".join(
                f"{old}->{new} x{count}" for (old, new), count in label_breakdown.changed_labels.items()
            )
            detail_parts.append(f"Relabeled: {changed_detail}")
        if detail_parts:
            self._log("; ".join(detail_parts))
        self.status_var.set(summary_msg)

    def _refresh_current_session(self) -> None:
        if self.selected_session is None:
            return
        self._load_session_data()



def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Raja annotation UI")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cvat-root", type=Path, default=DEFAULT_CVAT_ROOT)
    parser.add_argument("--sampling-rate", type=float, default=DEFAULT_SAMPLING_RATE)
    parser.add_argument(
        "--path-pairs",
        type=Path,
        default=DEFAULT_PATH_PAIR_CONFIG,
        help="Optional YAML file with selectable data_root/cvat_root pairs.",
    )
    parser.add_argument(
        "--pair-name",
        type=str,
        default=None,
        help="Name of the path pair to activate on startup.",
    )
    parser.add_argument(
        "--status-store",
        type=Path,
        default=None,
        help="Optional YAML file to persist FIF status selections.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO)

    pairs = load_path_pairs(args.path_pairs) if args.path_pairs else []
    cli_pair = PathPair("CLI provided", args.data_root, args.cvat_root)
    pairs.append(cli_pair)
    initial_pair = args.pair_name or pairs[0].name

    app = RajaAnnotationApp(
        args.data_root,
        args.cvat_root,
        sampling_rate=args.sampling_rate,
        path_pairs=pairs,
        selected_pair=initial_pair,
        status_store_path=args.status_store or DEFAULT_STATUS_STORE,
    )
    app.window.mainloop()


if __name__ == "__main__":
    main()
