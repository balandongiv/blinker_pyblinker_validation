"""Wrapper around the MNE browser used by the GUI."""

from __future__ import annotations

import math

import mne
from .session import AnnotationSession


PLACEHOLDER_LABELS = ("manual", "TO_MANUALLY_CHECK")


def _ensure_placeholder_labels(
    annotations: mne.Annotations, start: float
) -> tuple[mne.Annotations, dict[str, float]]:
    """Add zero-length annotations so common labels are available in the browser."""

    placeholder_onsets: dict[str, float] = {}
    annotations_for_plot = annotations

    for index, label in enumerate(PLACEHOLDER_LABELS):
        if label in annotations_for_plot.description:
            continue

        onset = start + (index * 1e-3)
        placeholder_onsets[label] = onset
        annotations_for_plot = annotations_for_plot + mne.Annotations(
            onset=[onset], duration=[0.0], description=[label]
        )

    return annotations_for_plot, placeholder_onsets


def launch_browser_and_collect(
    raw: mne.io.Raw,
    local_annotations: mne.Annotations,
    session: AnnotationSession,
    start: float,
    picks: list[str] | None = None,
    duration: float | None = None,
) -> mne.Annotations:
    """Open the MNE browser and return updated annotations aligned to global time."""

    status = "already annotated" if session.annotated_before else "new segment"
    title = (
        f"{session.fif_path.name} – Segment {session.start:.1f}–{session.end:.1f} s "
        f"– status: {status}"
    )
    print(f"Launching browser with title: {title}")

    raw_segment = raw.copy()
    annotations_for_plot, placeholder_onsets = _ensure_placeholder_labels(
        local_annotations, start
    )

    raw_segment.set_annotations(annotations_for_plot)
    plot_kwargs: dict[str, float | bool | list[str] | None | str] = {
        "title": title,
        "block": True,
        "start": start,
        "picks": picks,
            # "duration": duration
    }
    if duration is not None:
        plot_kwargs["duration"] = duration
    print(f"The duration for plotting is set to: {plot_kwargs.get('duration')}")
    raw_segment.plot(**plot_kwargs)
    ann = raw_segment.annotations

    if placeholder_onsets:
        keep_indices = [
            idx
            for idx, (desc, onset, duration) in enumerate(
                zip(ann.description, ann.onset, ann.duration, strict=True)
            )
            if not (
                desc in placeholder_onsets
                and math.isclose(duration, 0.0)
                and math.isclose(onset, placeholder_onsets[desc])
            )
        ]
        ann = ann[keep_indices]

    return ann
