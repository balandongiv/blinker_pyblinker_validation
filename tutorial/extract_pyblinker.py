"""Tutorial helper that runs pyblinker blink detection directly on FIF files."""

from __future__ import annotations

import logging
from pathlib import Path

from src.pyblinker import run_blinker_batch
from src.utils.config_utils import get_dataset_root, load_config

CONFIG_PATH = Path("../config/config.yaml")


def main() -> None:
    """Load configuration, resolve paths, and execute the pyblinker workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Loading configuration from %s", CONFIG_PATH)
    config = load_config(CONFIG_PATH)

    dataset_root = get_dataset_root(config)
    logging.info("Dataset root resolved to %s", dataset_root)

    project_root = Path(__file__).resolve().parents[1]
    logging.info("Project root resolved to %s", project_root)

    summary = run_blinker_batch(
        dataset_root=dataset_root,
        project_root=project_root,
        config=config,
    )

    output_root = summary["output_root"]
    logging.info("pyblinker outputs stored under %s", output_root)

    logging.info(
        "Blink detection completed for %s/%s FIF files",
        summary["processed"],
        summary["total"],
    )

    if summary["errors"]:
        logging.warning("Some files failed during blink detection")
        for path, message in summary["errors"].items():
            logging.error("%s -> %s", path, message)


if __name__ == "__main__":
    main()
