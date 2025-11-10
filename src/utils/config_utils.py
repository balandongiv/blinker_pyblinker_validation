"""Helper utilities for working with YAML configuration files."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and cache the YAML configuration file located at ``path``."""

    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle) or {}


def _as_mapping(config: Mapping[str, Any] | MutableMapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config, MutableMapping):
        return config
    return dict(config)


def _resolve_root_dir(config: Mapping[str, Any]) -> Path:
    paths = config.get("paths", {})
    root_dir = paths.get("root_dir")
    if root_dir is None:
        return REPO_ROOT

    root_path = Path(str(root_dir)).expanduser()
    if not root_path.is_absolute():
        root_path = (REPO_ROOT / root_path).resolve()
    return root_path


def _ensure_config_section(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    try:
        section_data = config[section]
    except KeyError as exc:  # pragma: no cover - guard clause
        raise KeyError(f"Configuration missing '{section}' section") from exc
    if not isinstance(section_data, Mapping):  # pragma: no cover - defensive
        raise TypeError(f"Configuration section '{section}' must be a mapping")
    return section_data


def resolve_path(
    config: Mapping[str, Any],
    key: str,
    *,
    section: str = "paths",
    base_dir: Path | None = None,
) -> Path:
    """Resolve ``key`` from ``config`` and return it as an absolute :class:`Path`."""

    config = _as_mapping(config)
    section_data = _ensure_config_section(config, section)

    try:
        raw_value = section_data[key]
    except KeyError as exc:  # pragma: no cover - guard clause
        raise KeyError(f"Configuration missing {section}.{key}.") from exc

    path = Path(str(raw_value)).expanduser()
    if path.is_absolute():
        return path

    root_dir = base_dir if base_dir is not None else _resolve_root_dir(config)
    return (root_dir / path).resolve()


def get_path_setting(
    config: Mapping[str, Any],
    key: str,
    *,
    env_var: str | None = None,
    default: Path | None = None,
    section: str = "paths",
    base_dir: Path | None = None,
) -> Path:
    """Return a path from ``config`` honouring an optional environment override."""

    if env_var and (override := os.environ.get(env_var)):
        return Path(override).expanduser()

    try:
        return resolve_path(config, key, section=section, base_dir=base_dir)
    except KeyError:
        if default is None:
            raise
        return default


def get_dataset_root(config: Mapping[str, Any], key: str = "raw_downsampled") -> Path:
    """Return the dataset root referenced by ``paths.<key>`` in ``config``."""

    return resolve_path(config, key, section="paths")


def get_default_channels(config: Mapping[str, Any]) -> Sequence[str] | None:
    """Return the default channel list defined in the configuration."""

    defaults = config.get("defaults", {})
    channels: Iterable[str] | None = defaults.get("channels")  # type: ignore[assignment]
    if channels is None:
        return None
    return tuple(str(ch).strip() for ch in channels if str(ch).strip())


def get_default_sampling_rate(config: Mapping[str, Any]) -> float | None:
    """Return the default sampling rate (Hz) defined in the configuration."""

    defaults = config.get("defaults", {})
    rate = defaults.get("sampling_rate_hz")
    if rate is None:
        return None
    try:
        return float(rate)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def get_default_blinker_plugin(config: Mapping[str, Any]) -> str | None:
    """Return the default EEGLAB Blinker plugin name from the configuration."""

    defaults = config.get("defaults", {})
    plugin = defaults.get("blinker_plugin")
    if plugin is None:
        return None
    return str(plugin)


