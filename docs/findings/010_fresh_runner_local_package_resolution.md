# Title
Fresh Runner Imported Installed Package Instead Of Local Workspace Code

# Date/time
2026-03-14 08:17:00 +08:00

# Hypothesis
The contradictory validation results are caused by `fresh_compare_from_csv.py` importing the installed `pyblinker` package from the conda environment when launched by file path, rather than importing the local workspace code under active development.

# Files inspected
- `blinker_pyblinker_validation/fresh_compare_from_csv.py`
- `pyblinker/blinker/pyblinker.py`
- `pyblinker/blinker/get_representative_channel.py`
- `pyblinker/pipeline_steps.py`
- `blinker_pyblinker_validation/blink_compare.py`
- `blinker_pyblinker_validation/finding/004_validate_top2_after_fix.py`
- `D:/dataset/murat_2018/9636595/exp01_pyblinker_results.pkl`
- `D:/dataset/murat_2018/9636595/exp02_pyblinker_results.pkl`
- `D:/dataset/murat_2018/12400382/exp02_pyblinker_results.pkl`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/getBlinkPositions.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/extractBlinks.m`

# Files changed
- `blinker_pyblinker_validation/fresh_compare_from_csv.py`
- `blinker_pyblinker_validation/finding/010_fresh_runner_local_package_resolution.md`

# Exact change made
- Inserted the repository root at the front of `sys.path` at the top of `fresh_compare_from_csv.py`, before any `pyblinker` imports.

# Why the change was made
- Running `python blinker_pyblinker_validation\\fresh_compare_from_csv.py ...` produced bad results (`9636595` returned `576` detections and `12400382` returned `1807` detections), while directly instantiating `BlinkDetector` from the workspace code on the same recordings returned the MATLAB-aligned counts (`557` and `658` respectively).
- This discrepancy is explained by import resolution: path-based execution set `sys.path[0]` to `blinker_pyblinker_validation`, which does not guarantee the workspace root is searched before the installed conda package.

# MATLAB reference used
- `utilities/getBlinkPositions.m`
- `utilities/extractBlinks.m`
- Stored MATLAB `blinker_results.pkl` `signalData` and `blinkFits` artifacts for subject-level comparisons

# Validation scope
- Focused tests: `pytest test/blinker_pyblinker_comparison -q`
- Fresh path-based validation before fix:
  - `exp02 top 2`
  - `exp02 bottom 2`
- Detector stage inspection:
  - `9636595`
  - `12400382`
- Fresh path-based validation after fix: pending in this log update

# Before/after metrics
- Before fix:
  - `exp02 top 2`: `9636571 = 99.2987377279`, `9636595 = 97.9699911739`
  - `exp02 bottom 2`: `12400388 = 0.4445855827`, `12400382 = 5.7606490872`
- After fix:
  - Pending rerun under a new experiment prefix

# Whether the change was kept or reverted
Kept

# Next recommended step
Rerun `top 2` and `bottom 2` with a new experiment prefix after the import-path fix. If those return to 100%, continue with `top/bottom 5`, `top/bottom 10`, then broader scaling.
