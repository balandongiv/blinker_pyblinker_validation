# 024 Validation Repository Restructure

1. Title
   Restructure `blinker_pyblinker_validation` into a reproducible, research-ready repository

2. Date/time
   2026-03-15 12:45:00 +08:00

3. Hypothesis
   The repository can be made substantially clearer and more reproducible by
   moving canonical validation code into `src/validation`, moving
   pipeline-facing scripts into `tutorial`, and documenting one canonical
   workflow for both datasets.

4. Files inspected
- `README.md`
- `good_practice.md`
- `src/validation/*.py`
- `src/ui_raja/annotation_import.py`
- `tutorial/murat_sequence/*.py`
- `tutorial/raja_sequence/*.py`
- `tutorial/murat_sequence/legacy/*.py`
- `pyproject.toml`
- `setup.py`
- `.gitignore`
- `mock_data/`

5. Files changed
- `src/validation/_paths.py`
- `src/validation/blink_compare.py`
- `src/validation/blink_compare_from_csv.py`
- `src/validation/fresh_compare_from_csv.py`
- `src/validation/fresh_compare_subjects.py`
- `src/validation/run_murat_full_with_status.py`
- `src/validation/run_murat_exp06_full_with_status.py`
- `src/ui_raja/annotation_import.py`
- `src/ui_raja/cvat_helpers.py`
- `tutorial/__init__.py`
- `tutorial/call_app_murat_ui.py`
- `tutorial/call_app_raja_ui.py`
- `tutorial/murat_sequence/__init__.py`
- `tutorial/murat_sequence/step0_download_dataset.py`
- `tutorial/murat_sequence/step1_prepare_dataset.py`
- `tutorial/murat_sequence/step2_run_blinker.py`
- `tutorial/murat_sequence/step3_validate_pyblinker.py`
- `tutorial/murat_sequence/complete_workflow_all_steps.py`
- `tutorial/raja_sequence/__init__.py`
- `tutorial/raja_sequence/helper.py`
- `tutorial/raja_sequence/step3_run_blinker.py`
- `tutorial/raja_sequence/step4_validate_pyblinker.py`
- `tutorial/murat_sequence/legacy/step4_compare_pyblinker_vs_blinker_legacy.py`
- `tutorial/murat_sequence/legacy/step5_compare_viz_vs_pyblinker_legacy.py`
- `tutorial/murat_sequence/legacy/step6_compare_viz_vs_blinker_legacy.py`
- `README.md`
- `good_practice.md`
- `pyproject.toml`
- `setup.py`
- `.gitignore`
- `reports/validation/.gitkeep`
- `mock_data/dataset/drowsy_driving_raja_processed/S13/S26_20190108_035218_3/ear_eog.fif`
- `mock_data/CVAT_visual_annotation/cvat_zip_final/S13/from_cvat/S26_20190108_035218_3.zip`

6. Exact change made
- Moved canonical validation code under `src/validation`.
- Added shared path helpers for report and source locations.
- Added tutorial wrappers for canonical Murat and driving-dataset validation runs.
- Updated moved validation modules to import from `src.validation` instead of the old vendored package path.
- Redirected Raja annotation helper logic into `src/ui_raja/cvat_helpers.py`.
- Fixed nested tutorial scripts so they resolve the repository root correctly from their new locations.
- Rewrote the Murat workflow orchestrator to reflect the canonical pipeline.
- Kept the old Murat comparison scripts under `tutorial/murat_sequence/legacy` and marked them as archived.
- Rewrote `README.md` and `good_practice.md` for the new structure.
- Updated packaging metadata so `src.*` and `tutorial.*` imports remain installable in editable mode.
- Redirected aggregate output expectations to `reports/validation/`.
- Added a tiny bundled Raja mock-data fixture so repository-local annotation
  import tests run without external datasets.

7. Why the change was made
- The previous layout mixed canonical code, experiments, and tutorial material.
- Validation modules still depended on a vendored copy of the old `pyblinker`
  repository layout.
- The public-facing documentation no longer matched how the repository actually worked.
- `murat_sequence/step4_compare_pyblinker_vs_blinker.py` represented an older
  workflow and was superseded by the fresh experiment-based validation runners.

8. MATLAB reference used
- None. This task was repository-structure and workflow documentation work, not
  detector logic debugging.

9. Validation scope
- Repository structure and import/path verification
- CLI smoke checks for moved tutorial and validation entry points
- Repository-local tests with bundled mock data
- No full dataset rerun performed as part of the restructure itself

10. Before/after metrics
- Not applicable for detector quality metrics in this iteration.
- Success criteria for this step are structural:
  - canonical code under `src`
  - tutorial entry points under `tutorial`
  - updated README and runbook
  - runnable imports from the new layout
  - local tests passing

11. Whether the change was kept or reverted
- Kept.

12. Step4 script decision
- `murat_sequence/step4_compare_pyblinker_vs_blinker.py` should not remain in
  the canonical pipeline.
- It was moved into `tutorial/murat_sequence/legacy/` and clearly marked as archived.
- The replacement workflow is:
  - `src.validation.fresh_compare_from_csv`
  - `src.validation.run_murat_full_with_status`
  - `tutorial/murat_sequence/step3_validate_pyblinker.py`

13. Next recommended step
- Run smoke verification on the new entry points.
- Remove the stale vendored `pyblinker/` directory from this repository once no
  remaining code depends on it.
- After that, rerun a small Murat and driving-dataset validation scope to
  confirm the restructure did not change behavior.

14. Verification performed
- `python -m py_compile ...` over the moved validation, UI, and tutorial entry points
- `python -m src.validation.fresh_compare_from_csv --help`
- `python -m src.validation.fresh_compare_subjects --help`
- `python -m src.validation.run_murat_full_with_status --help`
- `python tutorial/murat_sequence/step1_prepare_dataset.py --help`
- `python tutorial/murat_sequence/step2_run_blinker.py --help`
- `python tutorial/murat_sequence/step3_validate_pyblinker.py --help`
- `python tutorial/raja_sequence/step3_run_blinker.py --help`
- `python tutorial/raja_sequence/step4_validate_pyblinker.py --help`
- `pytest tests -q` -> `14 passed`

15. Status
- The stale vendored `pyblinker/` directory was removed after the migration.
