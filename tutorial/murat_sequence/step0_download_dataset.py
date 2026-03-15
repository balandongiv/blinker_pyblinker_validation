"""Download the full murat_2018 dataset listed in ``config/murat_2018_dataset.txt``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repository root (which contains the ``src`` package) is importable when
# this script is executed directly via ``python murat_sequence/step0_download_dataset.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.murat.download_dataset import DownloadError, download_dataset  # noqa: E402
from src.utils.config_utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    get_path_setting,
    load_config,
)
from src.utils.murat_dataset import resolve_dataset_file  # noqa: E402


CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_DATASET_FILE = get_path_setting(CONFIG, "dataset_file")
DEFAULT_ROOT = get_path_setting(CONFIG, "download_root", env_var="MURAT_DATASET_ROOT")
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="Text file listing dataset URLs. Defaults to config/murat_2018_dataset.txt.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Destination root for downloaded recordings.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per URL before giving up.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging. Omit to download all URLs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset_file = resolve_dataset_file(
        args.dataset_file,
        reference_dir=REPO_ROOT,
    )
    limit = None if args.limit is None or args.limit < 0 else args.limit

    try:
        count = download_dataset(
            dataset_file=dataset_file,
            root=args.root,
            limit=limit,
            retries=args.retries,
        )
    except DownloadError as exc:
        LOGGER.error("Dataset download failed: %s", exc)
        return 1

    LOGGER.info("Successfully processed %s file(s)", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
