"""Utilities for loading Raja data/CVAT path pairs from YAML."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PathPair:
    """A single selectable data/CVAT root pair."""

    name: str
    data_root: Path
    cvat_root: Path


def _resolve_path(value: Path, *, base_dir: Path) -> Path:
    """Resolve ``value`` relative to ``base_dir`` with a parent fallback."""

    if value.is_absolute():
        return value

    candidate = (base_dir / value).resolve()
    if candidate.exists():
        return candidate

    fallback = (base_dir.parent / value).resolve()
    if fallback.exists():
        return fallback

    return candidate


def _normalize_entry(entry: dict, *, base_dir: Path) -> PathPair | None:
    try:
        name = str(entry["name"])
        data_root = _resolve_path(Path(entry["data_root"]), base_dir=base_dir)
        cvat_root = _resolve_path(Path(entry["cvat_root"]), base_dir=base_dir)
    except Exception as exc:  # pragma: no cover - defensive parsing
        logger.warning("Skipping invalid path pair entry %s: %s", entry, exc)
        return None

    return PathPair(name=name, data_root=data_root, cvat_root=cvat_root)


def load_path_pairs(config_path: Path) -> List[PathPair]:
    """Load path pairs from a YAML config file."""

    if not config_path.exists():
        logger.info("Path pair config %s not found; falling back to CLI defaults.", config_path)
        return []

    try:
        content = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - defensive parsing
        logger.warning("Failed to read %s: %s", config_path, exc)
        return []

    base_dir = config_path.parent.resolve()
    pairs: list[PathPair] = []
    for entry in content.get("pairs", []):
        pair = _normalize_entry(entry, base_dir=base_dir)
        if pair is not None:
            pairs.append(pair)

    return pairs
