"""Logging helpers for the annotation UI."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_logger(log_path: Path) -> None:
    """Configure a file logger for annotation activity."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    existing_files = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path
    ]
    if not existing_files:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(file_handler)


def read_history(log_path: Path, max_lines: int = 50) -> list[str]:
    """Return the last ``max_lines`` of the history log if it exists."""

    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as log_file:
        lines = log_file.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def latest_file_history(log_path: Path, filename: str, max_entries: int = 3) -> list[str]:
    """Return the latest history entries that mention a specific file."""

    entries = read_history(log_path, max_lines=200)
    matched = [line for line in entries if filename in line]
    return matched[-max_entries:]


def latest_remark(log_path: Path, max_lines: int = 200) -> str | None:
    """Return the latest remark entry from the history log, if present."""

    for entry in reversed(read_history(log_path, max_lines=max_lines)):
        _, _, message = entry.partition(" - ")
        if message.startswith("Remark:"):
            return message
    return None


def last_edit_timestamp(csv_path: Path) -> str:
    """Return the last modification time for ``csv_path`` or ``"None"``."""

    if not csv_path.exists():
        return "None"
    modified = datetime.fromtimestamp(csv_path.stat().st_mtime)
    return modified.strftime("%Y-%m-%d %H:%M:%S")
