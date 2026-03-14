from __future__ import annotations

import argparse
from pathlib import Path

TARGET_EXTENSIONS = {".csv", ".edf", ".fif", ".pkl"}
TARGET_FILENAMES = {"pyblinker_results.json", "blinker_results.json"}
PROTECTED_EXTENSIONS = {".mat"}
DATASET_ROOT = Path(r"D:\dataset\murat_2018")


def collect_targets(root: Path) -> list[Path]:
    """Return files under root matching target extensions or target filenames, excluding protected extensions."""
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in PROTECTED_EXTENSIONS:
            continue

        name = path.name.lower()
        if suffix in TARGET_EXTENSIONS or name in TARGET_FILENAMES:
            matches.append(path)
    return matches


def delete_files(files: list[Path], perform_delete: bool) -> tuple[int, int]:
    deleted = 0
    failed = 0

    for file_path in files:
        if not perform_delete:
            print(f"[DRY RUN] Would delete: {file_path}")
            continue

        try:
            file_path.unlink()
            deleted += 1
            print(f"Deleted: {file_path}")
        except OSError as exc:
            failed += 1
            print(f"Failed: {file_path} ({exc})")

    return deleted, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find files with extensions .csv, .edf, .fif, .pkl "
            "and filenames pyblinker_results.json or blinker_results.json "
            "under D:\\dataset\\murat_2018 and remove them. "
            "MATLAB data files (.mat) are always preserved."
        )
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag, the script runs in dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = DATASET_ROOT.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"Invalid dataset directory: {root}")
        return 1

    files_to_remove = collect_targets(root)

    if not files_to_remove:
        print(f"No matching files found under: {root}")
        return 0

    print(f"Found {len(files_to_remove)} matching files under: {root}")
    deleted, failed = delete_files(files_to_remove, perform_delete=args.delete)

    if args.delete:
        print(f"Done. Deleted: {deleted}, Failed: {failed}")
        return 1 if failed else 0

    print("Dry run complete. Re-run with --delete to remove these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
