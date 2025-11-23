"""Wrapper around the MNE browser used by the GUI."""

from __future__ import annotations

import mne
from .session import AnnotationSession


def launch_browser_and_collect(
    raw: mne.io.Raw,
    local_annotations: mne.Annotations,
    session: AnnotationSession,
    start: float,
    picks: list[str] | None = None,
) -> mne.Annotations:
    """Open the MNE browser and return updated annotations aligned to global time."""

    status = "already annotated" if session.annotated_before else "new segment"
    title = (
        f"{session.fif_path.name} – Segment {session.start:.1f}–{session.end:.1f} s "
        f"– status: {status}"
    )
    print(f"Launching browser with title: {title}")

    raw_segment = raw.copy()
    annotations_for_plot = local_annotations
    placeholder_onset: float | None = None
    if "manual" not in annotations_for_plot.description:
        placeholder_onset = start
        annotations_for_plot = annotations_for_plot + mne.Annotations(
            onset=[placeholder_onset], duration=[0.0], description=["manual"]
        )

    raw_segment.set_annotations(annotations_for_plot)
    raw_segment.plot(title=title, block=True, start=start, picks=picks)
    ann = raw_segment.annotations

    if placeholder_onset is not None:
        keep_indices = [
            idx
            for idx, (desc, onset, duration) in enumerate(
                zip(ann.description, ann.onset, ann.duration, strict=True)
            )
            if not (
                desc == "manual"
                and duration == 0.0
                and onset == placeholder_onset
            )
        ]
        ann = ann[keep_indices]

    return ann
