"""Persistence helpers for Raja FIF status tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


def _relativize(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` when possible."""

    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


class StatusStore:
    """Load and save FIF status selections to a YAML file."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self._data: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            self._data = {}
            return

        try:
            self._data = yaml.safe_load(self.store_path.read_text()) or {}
        except Exception as exc:  # pragma: no cover - defensive parsing
            logger.warning("Failed to read status store %s: %s", self.store_path, exc)
            self._data = {}

    def load_for_root(self, data_root: Path) -> dict[Path, str]:
        """Return persisted statuses for ``data_root`` keyed by absolute FIF path."""

        root_key = str(data_root.resolve())
        saved = self._data.get(root_key, {})
        root_path = data_root.resolve()
        return {(root_path / Path(rel)).resolve(): status for rel, status in saved.items()}

    def update_status(self, data_root: Path, fif_path: Path, status: str) -> None:
        """Persist a single FIF status update immediately."""

        root_key = str(data_root.resolve())
        relative_key = _relativize(fif_path, data_root)
        root_map = self._data.setdefault(root_key, {})
        root_map[relative_key] = status
        self._save()

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(yaml.safe_dump(self._data))
