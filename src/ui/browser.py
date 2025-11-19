"""Wrapper around the MNE browser used by the GUI."""

from __future__ import annotations

import mne
import pandas as pd

from .annotation_io import frame_from_annotations
from .session import AnnotationSession


def launch_browser_and_collect(raw_segment: mne.io.Raw, session: AnnotationSession) -> pd.DataFrame:
    """Open the MNE browser and return updated annotations aligned to global time."""

    status = "already annotated" if session.annotated_before else "new segment"
    title = (
        f"{session.fif_path.name} – Segment {session.start:.1f}–{session.end:.1f} s "
        f"– status: {status}"
    )
    print(f"Launching browser with title: {title}")
    raw_segment.plot(title=title, block=True)
    return frame_from_annotations(raw_segment.annotations, base_time=float(session.start))
