"""Helper utilities for working with YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_dataset_root(config: Dict[str, Any], key: str = "raw_downsampled") -> Path:
    """Return the dataset root referenced by ``paths.<key>`` in ``config``."""

    try:
        base_path = config["paths"][key]
    except KeyError as exc:  # pragma: no cover - guard clause
        raise KeyError(
            f"Configuration missing paths.{key}."
        ) from exc

    return Path(base_path).expanduser()

