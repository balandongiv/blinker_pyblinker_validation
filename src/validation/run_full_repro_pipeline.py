from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import REPORTS_DIR, SUMMARY_METRICS_PATH
from src.validation.fresh_compare_subjects import DRIVING_SUBJECTS


TARGET_SHARE_PERCENT = 100.0


@dataclass(slots=True)
class StepResult:
    name: str
    command: list[str]
    started_at: str
    finished_at: str
    exit_code: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full end-to-end reproducibility pipeline: regenerate canonical "
            "inputs when needed, run MATLAB Blinker, execute the full PyBlinker "
            "validation sweeps for murat_2018 and driving_dataset, and write a "
            "combined manifest under reports/validation."
        ),
    )
    parser.add_argument(
        "--run-id",
        default="full_repro_v1",
        help=(
            "Logical experiment identifier used to derive Murat and driving prefixes, "
            "the pipeline log name, and the final manifest paths."
        ),
    )
    parser.add_argument("--murat-prefix", default=None, help="Override the Murat experiment prefix.")
    parser.add_argument("--driving-prefix", default=None, help="Override the driving experiment prefix.")
    parser.add_argument(
        "--murat-root",
        type=Path,
        default=Path(r"D:\dataset\murat_2018"),
        help="murat_2018 dataset root.",
    )
    parser.add_argument(
        "--driving-root",
        type=Path,
        default=Path(r"D:\dataset\drowsy_driving_raja_processed"),
        help="driving_dataset root.",
    )
    parser.add_argument(
        "--force-murat-prepare",
        action="store_true",
        help="Recreate Murat FIF/EDF files even when they already exist.",
    )
    parser.add_argument(
        "--force-matlab",
        action="store_true",
        help="Overwrite existing MATLAB Blinker outputs for both datasets.",
    )
    parser.add_argument(
        "--force-validation",
        action="store_true",
        help="Overwrite existing prefixed PyBlinker outputs for both datasets.",
    )
    parser.add_argument(
        "--skip-murat-prepare",
        action="store_true",
        help="Skip Murat MAT -> FIF/EDF preparation.",
    )
    parser.add_argument(
        "--skip-murat-blinker",
        action="store_true",
        help="Skip the Murat MATLAB Blinker run.",
    )
    parser.add_argument(
        "--skip-driving-blinker",
        action="store_true",
        help="Skip the driving-dataset MATLAB Blinker run.",
    )
    parser.add_argument(
        "--skip-murat-validation",
        action="store_true",
        help="Skip the Murat PyBlinker validation sweep.",
    )
    parser.add_argument(
        "--skip-driving-validation",
        action="store_true",
        help="Skip the driving-dataset PyBlinker validation sweep.",
    )
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for child commands.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _all_murat_recording_ids(csv_path: Path) -> list[str]:
    summary = pd.read_csv(csv_path, dtype={"recording_id": "string"})
    if "recording_id" not in summary.columns:
        raise KeyError(f"Column 'recording_id' not found in {csv_path}")
    recording_ids = summary["recording_id"].dropna().astype("string").str.strip()
    recording_ids = recording_ids[recording_ids != ""]
    return recording_ids.tolist()


def _pyblinker_locations() -> list[str]:
    spec = importlib.util.find_spec("pyblinker")
    if spec is None:
        return []
    if spec.submodule_search_locations:
        return [str(Path(path)) for path in spec.submodule_search_locations]
    if spec.origin:
        return [str(Path(spec.origin))]
    return []


def _run_step(step_name: str, command: list[str], log_handle) -> StepResult:
    started_at = _utc_now()
    header = f"\n===== [{started_at}] START {step_name} =====\nCOMMAND: {' '.join(command)}\n"
    print(header, end="")
    log_handle.write(header)
    log_handle.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    def _console_write(text: str) -> None:
        try:
            print(text, end="")
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf8"
            sys.stdout.buffer.write(text.encode(encoding, errors="replace"))
            sys.stdout.buffer.flush()

    assert process.stdout is not None
    for line in process.stdout:
        _console_write(line)
        log_handle.write(line)
    process.wait()
    log_handle.flush()

    finished_at = _utc_now()
    footer = (
        f"===== [{finished_at}] END {step_name} (exit_code={process.returncode}) =====\n"
    )
    print(footer, end="")
    log_handle.write(footer)
    log_handle.flush()

    return StepResult(
        name=step_name,
        command=command,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=int(process.returncode),
    )


def _is_clean_share(value: object, target: float = TARGET_SHARE_PERCENT) -> bool:
    try:
        share = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(share) and math.isclose(share, target, rel_tol=0.0, abs_tol=1e-9)


def _summarise_validation(summary_path: Path, overall_path: Path, dataset_name: str) -> dict[str, object]:
    summary = pd.read_csv(summary_path)
    overall = json.loads(overall_path.read_text(encoding="utf8"))

    if "share_within_tolerance_percent" not in summary.columns:
        raise KeyError(f"share_within_tolerance_percent not found in {summary_path}")

    poor_rows = summary[~summary["share_within_tolerance_percent"].map(_is_clean_share)].copy()
    poor_preview = []
    if not poor_rows.empty:
        preview_cols = ["recording_id", "share_within_tolerance_percent"]
        poor_preview = poor_rows.loc[:, [col for col in preview_cols if col in poor_rows.columns]].to_dict("records")

    min_share = None
    if not summary.empty:
        min_share = float(pd.to_numeric(summary["share_within_tolerance_percent"], errors="coerce").min())

    return {
        "dataset": dataset_name,
        "summary_path": str(summary_path),
        "overall_path": str(overall_path),
        "recording_count": int(len(summary)),
        "all_share_100": poor_rows.empty,
        "min_share_within_tolerance_percent": min_share,
        "poor_recordings": poor_preview,
        "overall_metrics": overall,
    }


def _write_manifest_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def _write_manifest_md(path: Path, payload: dict[str, object]) -> None:
    murat = payload.get("murat_validation", {})
    driving = payload.get("driving_validation", {})
    lines = [
        f"# {payload['run_id']} Full Reproducibility Pipeline",
        "",
        f"- started_at: {payload['started_at']}",
        f"- finished_at: {payload['finished_at']}",
        f"- python_exe: {payload['python_exe']}",
        f"- pyblinker_locations: {', '.join(payload.get('pyblinker_locations', [])) or 'not found'}",
        f"- pipeline_log: {payload['pipeline_log_path']}",
        "",
        "## Prefixes",
        f"- murat_prefix: {payload['murat_prefix']}",
        f"- driving_prefix: {payload['driving_prefix']}",
        "",
        "## Step Results",
    ]
    for step in payload.get("steps", []):
        lines.append(
            f"- {step['name']}: exit_code={step['exit_code']}, started={step['started_at']}, finished={step['finished_at']}"
        )

    lines.extend(
        [
            "",
            "## Murat Validation",
            f"- summary_path: {murat.get('summary_path', 'skipped')}",
            f"- overall_path: {murat.get('overall_path', 'skipped')}",
            f"- recording_count: {murat.get('recording_count', 'n/a')}",
            f"- all_share_100: {murat.get('all_share_100', 'n/a')}",
            f"- min_share_within_tolerance_percent: {murat.get('min_share_within_tolerance_percent', 'n/a')}",
            "",
            "## Driving Validation",
            f"- summary_path: {driving.get('summary_path', 'skipped')}",
            f"- overall_path: {driving.get('overall_path', 'skipped')}",
            f"- recording_count: {driving.get('recording_count', 'n/a')}",
            f"- all_share_100: {driving.get('all_share_100', 'n/a')}",
            f"- min_share_within_tolerance_percent: {driving.get('min_share_within_tolerance_percent', 'n/a')}",
        ]
    )

    if murat.get("poor_recordings"):
        lines.extend(["", "## Murat Non-100 Recordings"])
        lines.extend(
            f"- {row['recording_id']}: {row['share_within_tolerance_percent']}"
            for row in murat["poor_recordings"]
        )

    if driving.get("poor_recordings"):
        lines.extend(["", "## Driving Non-100 Recordings"])
        lines.extend(
            f"- {row['recording_id']}: {row['share_within_tolerance_percent']}"
            for row in driving["poor_recordings"]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _child_command(python_exe: Path, script_path: str, *args: str) -> list[str]:
    return [str(python_exe), "-u", script_path, *args]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    murat_recording_ids = _all_murat_recording_ids(SUMMARY_METRICS_PATH)
    murat_prefix = args.murat_prefix or f"{args.run_id}_murat"
    driving_prefix = args.driving_prefix or f"{args.run_id}_driving"
    log_path = REPORTS_DIR / f"{args.run_id}_pipeline.log"
    manifest_json_path = REPORTS_DIR / f"{args.run_id}_pipeline_manifest.json"
    manifest_md_path = REPORTS_DIR / f"{args.run_id}_pipeline_manifest.md"

    payload: dict[str, object] = {
        "run_id": args.run_id,
        "started_at": _utc_now(),
        "finished_at": None,
        "python_exe": str(args.python_exe),
        "pyblinker_locations": _pyblinker_locations(),
        "pipeline_log_path": str(log_path),
        "manifest_json_path": str(manifest_json_path),
        "manifest_md_path": str(manifest_md_path),
        "murat_prefix": murat_prefix,
        "driving_prefix": driving_prefix,
        "murat_root": str(args.murat_root),
        "driving_root": str(args.driving_root),
        "steps": [],
        "murat_validation": {},
        "driving_validation": {},
    }

    step_commands: list[tuple[str, list[str]]] = []
    if not args.skip_murat_prepare:
        prepare_args = ["tutorial/murat_sequence/step1_prepare_dataset.py", "--root", str(args.murat_root), "--skip-download"]
        if args.force_murat_prepare:
            prepare_args.append("--force")
        step_commands.append(("murat_prepare", _child_command(args.python_exe, *prepare_args)))

    if not args.skip_murat_blinker:
        murat_blinker_args = ["tutorial/murat_sequence/step2_run_blinker.py", "--root", str(args.murat_root)]
        if args.force_matlab:
            murat_blinker_args.append("--force")
        step_commands.append(("murat_blinker", _child_command(args.python_exe, *murat_blinker_args)))

    if not args.skip_driving_blinker:
        driving_blinker_args = ["tutorial/raja_sequence/step3_run_blinker.py", "--root", str(args.driving_root)]
        if args.force_matlab:
            driving_blinker_args.append("--force")
        step_commands.append(("driving_blinker", _child_command(args.python_exe, *driving_blinker_args)))

    if not args.skip_murat_validation:
        murat_validate_args = [
            "tutorial/murat_sequence/step3_validate_pyblinker.py",
            "--prefix",
            murat_prefix,
            "--selection",
            "top",
            "--n",
            str(len(murat_recording_ids)),
        ]
        if args.force_validation:
            murat_validate_args.append("--force-rerun")
        step_commands.append(("murat_validation", _child_command(args.python_exe, *murat_validate_args)))

    if not args.skip_driving_validation:
        driving_validate_args = [
            "tutorial/raja_sequence/step4_validate_pyblinker.py",
            "--dataset",
            "driving_dataset",
            "--prefix",
            driving_prefix,
            "--subjects",
            ",".join(DRIVING_SUBJECTS),
            "--restrict-py-to-comparison-channels",
            "--continue-on-failure",
        ]
        if args.force_validation:
            driving_validate_args.append("--force-rerun")
        step_commands.append(("driving_validation", _child_command(args.python_exe, *driving_validate_args)))

    exit_code = 0
    with log_path.open("w", encoding="utf8") as log_handle:
        for step_name, command in step_commands:
            result = _run_step(step_name, command, log_handle)
            payload["steps"].append(asdict(result))
            _write_manifest_json(manifest_json_path, payload)
            _write_manifest_md(manifest_md_path, payload)
            if result.exit_code != 0:
                exit_code = result.exit_code
                break

    if exit_code == 0 and not args.skip_murat_validation:
        murat_summary_path = REPORTS_DIR / f"{murat_prefix}_top{len(murat_recording_ids)}_summary.csv"
        murat_overall_path = REPORTS_DIR / f"{murat_prefix}_top{len(murat_recording_ids)}_overall.json"
        payload["murat_validation"] = _summarise_validation(
            murat_summary_path,
            murat_overall_path,
            "murat_2018",
        )
        if not payload["murat_validation"]["all_share_100"]:
            exit_code = 1

    if exit_code == 0 and not args.skip_driving_validation:
        driving_summary_path = REPORTS_DIR / f"{driving_prefix}_driving_dataset_{len(DRIVING_SUBJECTS)}subjects_summary.csv"
        driving_overall_path = REPORTS_DIR / f"{driving_prefix}_driving_dataset_{len(DRIVING_SUBJECTS)}subjects_overall.json"
        payload["driving_validation"] = _summarise_validation(
            driving_summary_path,
            driving_overall_path,
            "driving_dataset",
        )
        if not payload["driving_validation"]["all_share_100"]:
            exit_code = 1

    payload["finished_at"] = _utc_now()
    _write_manifest_json(manifest_json_path, payload)
    _write_manifest_md(manifest_md_path, payload)

    print()
    print(f"Pipeline log: {log_path}")
    print(f"Pipeline manifest JSON: {manifest_json_path}")
    print(f"Pipeline manifest Markdown: {manifest_md_path}")
    if payload.get("murat_validation"):
        print(
            "Murat all_share_100 = "
            f"{payload['murat_validation'].get('all_share_100')} "
            f"(summary: {payload['murat_validation'].get('summary_path')})"
        )
    if payload.get("driving_validation"):
        print(
            "Driving all_share_100 = "
            f"{payload['driving_validation'].get('all_share_100')} "
            f"(summary: {payload['driving_validation'].get('summary_path')})"
        )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
