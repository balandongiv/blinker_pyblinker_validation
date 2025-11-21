"""Wrapper around the MNE browser used by the GUI."""

from __future__ import annotations

import mne
from .session import AnnotationSession


def launch_browser_and_collect(
    raw: mne.io.Raw,
    local_annotations: mne.Annotations,
    session: AnnotationSession,
    start: float,
) -> mne.Annotations:
    """Open the MNE browser and return updated annotations aligned to global time."""

    status = "already annotated" if session.annotated_before else "new segment"
    title = (
        f"{session.fif_path.name} – Segment {session.start:.1f}–{session.end:.1f} s "
        f"– status: {status}"
    )
    print(f"Launching browser with title: {title}")


    raw_segment = raw.copy()
    raw_segment.set_annotations(local_annotations)
    raw_segment.plot(title=title, block=True, start=start)
    ann = raw_segment.annotations
    return ann
