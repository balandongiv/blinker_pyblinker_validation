"""Discovery helpers for Raja dataset structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .constants import PRIMARY_FIF_CANDIDATES


@dataclass
class SessionInfo:
    """Container describing a single Raja recording session."""

    subject_id: str
    session_name: str
    fif_path: Path

    @property
    def annotation_csv(self) -> Path:
        """Return the derived annotation CSV path for this session."""

        return self.fif_path.with_suffix(".csv")

    @property
    def backups_dir(self) -> Path:
        """Directory for annotation backups alongside the FIF file."""

        return self.fif_path.parent / "backups"


class RajaDataset:
    """Index of available Raja sessions grouped by subject."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.sessions_by_subject: Dict[str, List[SessionInfo]] = {}
        self._discover()

    def _discover(self) -> None:
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"The data root {self.data_root} does not exist; please check your path."
            )

        for subject_dir in sorted(
            p for p in self.data_root.iterdir() if p.is_dir() and p.name.startswith("S")
        ):
            sessions = []
            for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                fif_path = self._pick_fif_file(session_dir)
                if fif_path is None:
                    continue
                sessions.append(
                    SessionInfo(
                        subject_id=subject_dir.name,
                        session_name=session_dir.name,
                        fif_path=fif_path,
                    )
                )
            if sessions:
                self.sessions_by_subject[subject_dir.name] = sessions

        if not self.sessions_by_subject:
            raise FileNotFoundError("No FIF files were found in the Raja dataset root.")

    def subjects(self) -> list[str]:
        return sorted(self.sessions_by_subject)

    def sessions_for(self, subject_id: str) -> list[SessionInfo]:
        return list(self.sessions_by_subject.get(subject_id, []))

    def all_sessions(self) -> list[SessionInfo]:
        return [session for sessions in self.sessions_by_subject.values() for session in sessions]

    def _pick_fif_file(self, session_dir: Path) -> Path | None:
        for name in PRIMARY_FIF_CANDIDATES:
            candidate = session_dir / name
            if candidate.exists():
                return candidate
        fif_files = sorted(session_dir.glob("*.fif")) + sorted(session_dir.glob("*.fif.gz"))
        return fif_files[0] if fif_files else None


def flatten_sessions(dataset: RajaDataset, subjects: Iterable[str] | None = None) -> list[SessionInfo]:
    """Return sessions, optionally filtered to a subset of subjects."""

    if subjects is None:
        return dataset.all_sessions()
    allowed = set(subjects)
    return [session for session in dataset.all_sessions() if session.subject_id in allowed]
