"""Batch-create and review blink ground-truth annotations for Murat et al. (2018).

This tutorial step walks through comparing PyBlinker detections against the
original MATLAB Blinker outputs for every recording under
``data/murat_2018``.  For each recording the helper utilities load the
``pyblinker_results.pkl`` and ``blinker_results.pkl`` payloads, compute
alignment metrics, attach comparison annotations to the corresponding
``.fif`` raw file, and automatically open the interactive MNE browser so the
annotations can be inspected or edited.  Recordings that already include an
``*_annot_inspected.csv`` file are skipped during batch runs unless the
``--include-inspected`` flag is provided.  You can also target a specific
recording by passing one or more ``--recording-id`` values or explicit file
paths when you want to re-run the comparison for a single subject.
"""

from __future__ import annotations

from src.utils.ground_truth import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
