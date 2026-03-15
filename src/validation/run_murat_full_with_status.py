from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import REPORTS_DIR, SUMMARY_METRICS_PATH
from src.validation.fresh_compare_from_csv import (
    DATASET_ROOT,
    TOLERANCE_SAMPLES,
    _order_summary,
    _write_experiment_outputs,
    load_selected_recording_ids,
    process_recording,
)
from src.validation.stat import build_overall_summary, build_summary_frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full Murat ordered sweep with live status files, a rolling log, "
            "and fresh PyBlinker results written beside each dataset recording."
        ),
    )
    parser.add_argument("--prefix", default="exp06", help="Experiment prefix.")
    parser.add_argument(
        "--selection",
        choices=("top", "bottom"),
        default="top",
        help="Whether to start from the top or bottom of summary_metrics.csv.",
    )
    parser.add_argument("--n", type=int, default=74, help="Number of ordered recordings to process.")
    parser.add_argument("--csv-path", type=Path, default=SUMMARY_METRICS_PATH, help="Ordered summary_metrics.csv path.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT, help="murat_2018 dataset root.")
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=TOLERANCE_SAMPLES,
        help="Tolerance window used for comparison metrics.",
    )
    parser.add_argument(
        "--target-share-percent",
        type=float,
        default=100.0,
        help="Expected share_within_tolerance_percent for a clean run.",
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=10, help="Status heartbeat interval.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
        help="Maximum worker threads.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Regenerate outputs even if they exist.")
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _configure_logger(log_path: Path) -> logging.Logger:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"validation.full_sweep.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _write_status(
    *,
    prefix: str,
    selection: str,
    started_at: str,
    recording_ids: list[str],
    completed: list[dict[str, object]],
    in_progress: list[str],
    failed: list[dict[str, object]],
    max_workers: int,
    log_path: Path,
    status_json_path: Path,
    status_md_path: Path,
    logger: logging.Logger,
    finished: bool,
) -> None:
    total = len(recording_ids)
    completed_ids = {entry["recording_id"] for entry in completed}
    pending_ids = [
        recording_id
        for recording_id in recording_ids
        if recording_id not in completed_ids and recording_id not in set(in_progress)
    ]
    payload = {
        "prefix": prefix,
        "dataset": "murat_2018",
        "selection": selection,
        "recording_count": total,
        "started_at": started_at,
        "last_heartbeat": _utc_now(),
        "finished": finished,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "in_progress_count": len(in_progress),
        "pending_count": len(pending_ids),
        "max_workers": max_workers,
        "completed": completed,
        "in_progress": in_progress,
        "pending_preview": pending_ids[:10],
        "failed": failed,
        "log_path": str(log_path),
    }
    status_json_path.write_text(json.dumps(payload, indent=2), encoding="utf8")

    heading = f"# {prefix} {selection}{total} live status"
    lines = [
        heading,
        "",
        f"- started_at: {started_at}",
        f"- last_heartbeat: {payload['last_heartbeat']}",
        f"- finished: {finished}",
        f"- total: {total}",
        f"- completed: {len(completed)}",
        f"- in_progress: {len(in_progress)}",
        f"- pending: {len(pending_ids)}",
        f"- failed: {len(failed)}",
        f"- max_workers: {max_workers}",
        f"- log: {log_path}",
        "",
        "## In Progress",
    ]
    if in_progress:
        lines.extend(f"- {recording_id}" for recording_id in in_progress)
    else:
        lines.append("- none")
    lines.extend(["", "## Recent Completed"])
    if completed:
        lines.extend(
            f"- {entry['recording_id']}: share={entry['share_within_tolerance_percent']}, status={entry['artifact_status']}"
            for entry in completed[-10:]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Failed"])
    if failed:
        lines.extend(f"- {entry['recording_id']}: {entry['error']}" for entry in failed[-10:])
    else:
        lines.append("- none")
    status_md_path.write_text("\n".join(lines) + "\n", encoding="utf8")

    logger.info(
        "heartbeat finished=%s completed=%s in_progress=%s pending=%s failed=%s",
        finished,
        len(completed),
        len(in_progress),
        len(pending_ids),
        len(failed),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recording_ids = load_selected_recording_ids(args.csv_path, n_rows=args.n, selection=args.selection)
    run_label = f"{args.selection}{len(recording_ids)}"
    status_json_path = REPORTS_DIR / f"{args.prefix}_{run_label}_live_status.json"
    status_md_path = REPORTS_DIR / f"{args.prefix}_{run_label}_live_status.md"
    log_path = REPORTS_DIR / f"{args.prefix}_{run_label}_live_log.txt"
    logger = _configure_logger(log_path)

    started_at = _utc_now()
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    results = []

    logger.info(
        "starting full sweep prefix=%s dataset_root=%s recordings=%s max_workers=%s force_rerun=%s",
        args.prefix,
        args.dataset_root,
        len(recording_ids),
        args.max_workers,
        args.force_rerun,
    )
    _write_status(
        prefix=args.prefix,
        selection=args.selection,
        started_at=started_at,
        recording_ids=recording_ids,
        completed=completed,
        in_progress=[],
        failed=failed,
        max_workers=args.max_workers,
        log_path=log_path,
        status_json_path=status_json_path,
        status_md_path=status_md_path,
        logger=logger,
        finished=False,
    )

    future_to_recording: dict[object, str] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for recording_id in recording_ids:
            future = executor.submit(
                process_recording,
                recording_id,
                dataset_root=args.dataset_root,
                prefix=args.prefix,
                tolerance_samples=args.tolerance_samples,
                plot=False,
                target_share_percent=args.target_share_percent,
                force_rerun=args.force_rerun,
            )
            future_to_recording[future] = recording_id

        pending_futures = set(future_to_recording)
        while pending_futures:
            done, pending_futures = wait(
                pending_futures,
                timeout=args.heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            in_progress = [future_to_recording[future] for future in pending_futures]

            if not done:
                _write_status(
                    prefix=args.prefix,
                    selection=args.selection,
                    started_at=started_at,
                    recording_ids=recording_ids,
                    completed=completed,
                    in_progress=in_progress,
                    failed=failed,
                    max_workers=args.max_workers,
                    log_path=log_path,
                    status_json_path=status_json_path,
                    status_md_path=status_md_path,
                    logger=logger,
                    finished=False,
                )
                continue

            for future in done:
                recording_id = future_to_recording[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - operational path
                    failed.append({"recording_id": recording_id, "error": repr(exc)})
                    logger.exception("recording failed: %s", recording_id)
                else:
                    results.append(result)
                    share = float(result.comparison.metrics.get("share_within_tolerance_percent"))
                    completed.append(
                        {
                            "recording_id": recording_id,
                            "share_within_tolerance_percent": share,
                            "artifact_status": result.artifact_status,
                        }
                    )
                    logger.info(
                        "completed recording=%s share=%s status=%s",
                        recording_id,
                        share,
                        result.artifact_status,
                    )

            _write_status(
                prefix=args.prefix,
                selection=args.selection,
                started_at=started_at,
                recording_ids=recording_ids,
                completed=completed,
                in_progress=in_progress,
                failed=failed,
                max_workers=args.max_workers,
                log_path=log_path,
                status_json_path=status_json_path,
                status_md_path=status_md_path,
                logger=logger,
                finished=False,
            )

    ordered_summary = _order_summary(build_summary_frame([result.comparison for result in results]), recording_ids)
    if ordered_summary.empty and "recording_id" not in ordered_summary.columns:
        ordered_summary = pd.DataFrame({"recording_id": pd.Series(dtype="string")})
    ordered_summary["artifact_status"] = ordered_summary["recording_id"].map(
        {result.recording_id: result.artifact_status for result in results}
    )
    ordered_summary["result_file"] = f"{args.prefix}_pyblinker_results.pkl"
    ordered_summary["selection"] = args.selection

    overall = build_overall_summary(ordered_summary)
    summary_path, overall_path, selection_path = _write_experiment_outputs(
        summary=ordered_summary,
        overall=overall,
        prefix=args.prefix,
        n_subjects=len(recording_ids),
        selection=args.selection,
        recording_ids=recording_ids,
    )
    logger.info("summary csv=%s", summary_path)
    logger.info("summary json=%s", overall_path)
    logger.info("selection csv=%s", selection_path)

    _write_status(
        prefix=args.prefix,
        selection=args.selection,
        started_at=started_at,
        recording_ids=recording_ids,
        completed=completed,
        in_progress=[],
        failed=failed,
        max_workers=args.max_workers,
        log_path=log_path,
        status_json_path=status_json_path,
        status_md_path=status_md_path,
        logger=logger,
        finished=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
