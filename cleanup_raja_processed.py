from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_ROOT = Path(r"D:\dataset\drowsy_driving_raja_processed")
CHILD_FOLDER_NAME = "blinker_pyblinker_validation"
DEFAULT_FILENAMES = {"pyblinker_results.json", "blinker_results.json","blinker_results.pkl"}


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def discover_default_scan_roots(root: Path) -> list[Path]:
    scan_roots: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        candidate = child / CHILD_FOLDER_NAME
        if candidate.is_dir():
            scan_roots.append(candidate.resolve())
    return scan_roots


def resolve_scan_roots(root: Path, child_folders: list[str]) -> list[Path]:
    if not child_folders:
        return discover_default_scan_roots(root)

    selected = child_folders
    scan_roots: list[Path] = []

    for child in selected:
        candidate = Path(child)
        if not candidate.is_absolute():
            candidate = root / candidate

        candidate = candidate.expanduser().resolve()
        if not _inside_root(candidate, root):
            print(f"Skipping outside root: {candidate}")
            continue

        if not candidate.exists() or not candidate.is_dir():
            print(f"Skipping missing folder: {candidate}")
            continue

        scan_roots.append(candidate)

    return scan_roots


def normalize_extensions(ext_values: list[str]) -> set[str]:
    normalized: set[str] = set()
    for ext in ext_values:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def should_delete(
    path: Path,
    all_files: bool,
    filenames: set[str],
    extensions: set[str],
    patterns: list[str],
) -> bool:
    if all_files:
        return True

    name = path.name.lower()
    suffix = path.suffix.lower()

    if name in filenames:
        return True
    if suffix in extensions:
        return True

    path_text = path.as_posix().lower()
    for pattern in patterns:
        pat = pattern.lower()
        if fnmatch(name, pat) or fnmatch(path_text, pat):
            return True

    return False


def collect_targets(
    scan_roots: list[Path],
    all_files: bool,
    filenames: set[str],
    extensions: set[str],
    patterns: list[str],
) -> list[Path]:
    matches: list[Path] = []

    for scan_root in scan_roots:
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue

            if should_delete(path, all_files, filenames, extensions, patterns):
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
            "Recursively clean files under selected child folders of "
            "D:\\dataset\\drowsy_driving_raja_processed. "
            "Dry-run by default; add --delete to actually remove files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder to scan (default: D:\\dataset\\drowsy_driving_raja_processed).",
    )
    parser.add_argument(
        "--child",
        action="append",
        default=[],
        help=(
            "Relative or absolute child folder to scan. Repeat for multiple folders. "
            "If omitted, all immediate subfolders containing "
            "blinker_pyblinker_validation are scanned (e.g., S2/S3/S4...)."
        ),
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help=(
            "Delete every file under selected child folders. This is already the "
            "default when no --filename/--ext/--pattern filter is provided."
        ),
    )
    parser.add_argument(
        "--filename",
        action="append",
        default=[],
        help=(
            "Target filename (exact match). Repeat for multiple values, e.g. "
            "--filename blinker_results.json --filename blinker_results.pkl"
        ),
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[],
        help="Target extension (with or without dot), e.g. --ext .pkl or --ext pkl",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help=(
            "Glob pattern for filename or full path, e.g. --pattern *.json "
            "or --pattern *full_repro*.pkl"
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag, the script runs in dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"Invalid root directory: {root}")
        return 1

    scan_roots = resolve_scan_roots(root, args.child)
    if not scan_roots:
        print("No valid child folders found/selected. Nothing to do.")
        return 1

    filenames = {name.lower() for name in args.filename if name.strip()}
    extensions = normalize_extensions(args.ext)
    patterns = [p for p in args.pattern if p.strip()]

    effective_all_files = args.all_files or (
        not filenames and not extensions and not patterns
    )

    print(f"Root: {root}")
    print("Scanning child folders:")
    for folder in scan_roots:
        print(f"- {folder}")

    if effective_all_files:
        print("Mode: delete all files under selected child folders")
    else:
        print(
            "Mode: filtered delete "
            f"(filenames={sorted(filenames)}, extensions={sorted(extensions)}, patterns={patterns})"
        )

    files_to_remove = collect_targets(
        scan_roots=scan_roots,
        all_files=effective_all_files,
        filenames=filenames,
        extensions=extensions,
        patterns=patterns,
    )

    if not files_to_remove:
        print("No matching files found.")
        return 0

    print(f"Found {len(files_to_remove)} matching files.")
    deleted, failed = delete_files(files_to_remove, perform_delete=args.delete)

    if args.delete:
        print(f"Done. Deleted: {deleted}, Failed: {failed}")
        return 1 if failed else 0

    print("Dry run complete. Re-run with --delete to remove these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

