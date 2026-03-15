"""Backward-compatible wrapper for the Murat full-sweep runner."""

from __future__ import annotations

from src.validation.run_murat_full_with_status import main


if __name__ == "__main__":
    raise SystemExit(main())

