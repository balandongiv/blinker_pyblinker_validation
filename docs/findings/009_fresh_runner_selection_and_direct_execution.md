# Title
Fresh Runner Selection And Direct Execution Support

# Date/time
2026-03-14 07:59:17 +08:00

# Hypothesis
The validation workflow is being slowed or blocked by the runner rather than by blink-detection logic. In particular, `fresh_compare_from_csv.py` needs explicit top/bottom selection, experiment-scoped output naming, and a direct-script import fix so fresh reruns can be executed reliably without accidentally reusing stale artifacts.

# Files inspected
- `blinker_pyblinker_validation/fresh_compare_from_csv.py`
- `blinker_pyblinker_validation/blink_compare_from_csv.py`
- `blinker_pyblinker_validation/blink_compare.py`
- `blinker_pyblinker_validation/stat.py`
- `blinker_pyblinker_validation/summary_metrics.csv`

# Files changed
- `blinker_pyblinker_validation/fresh_compare_from_csv.py`
- `blinker_pyblinker_validation/finding/009_fresh_runner_selection_and_direct_execution.md`

# Exact change made
- Added `--selection {top,bottom}` to `fresh_compare_from_csv.py`.
- Added local CSV slicing logic inside `fresh_compare_from_csv.py` so the runner no longer depends on the top-only helper from `blink_compare_from_csv.py`.
- Updated experiment output naming to include the selection label and added a recording-list CSV for reproducibility.
- Fixed direct-script execution by importing the local validation `stat.py` via `importlib.util` instead of falling through to Python's stdlib `stat` module.
- Kept plot control via the existing `--plot` flag and preserved the thread-vs-single execution policy, with `auto` switching to threads when more than 10 recordings are selected.

# Why the change was made
The user requested bottom-up validation from the same runner, optional visualization, and experiment isolation whenever reruns happen. The direct execution import bug would also make `python blinker_pyblinker_validation\fresh_compare_from_csv.py` fail even though `python -m ...` works.

# MATLAB reference used
- None for this runner-only change. No detector logic was changed.

# Validation scope
- Script compilation: `python -m py_compile blinker_pyblinker_validation\fresh_compare_from_csv.py`
- Module help: `python -m blinker_pyblinker_validation.fresh_compare_from_csv --help`
- Direct-script help: `python blinker_pyblinker_validation\fresh_compare_from_csv.py --help`
- Subject validation: pending
- Tests: pending

# Before/after metrics
- Before runner change: no new subject metrics captured in this log yet.
- After runner change: detector metrics pending; runner compile/help checks pass for both module and direct-script execution.

# Whether the change was kept or reverted
Kept

# Next recommended step
Run fresh staged validations with a new experiment prefix for bottom 2, bottom 5, bottom 10, and recheck top 2, top 5, and top 10 under the same prefix. If any group is below 100%, investigate the first divergence before scaling further.
