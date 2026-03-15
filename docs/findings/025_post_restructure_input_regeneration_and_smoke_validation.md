# 025 Post-Restructure Input Regeneration And Smoke Validation

1. Title
   Regenerate canonical MATLAB-backed validation inputs after the repository restructure and verify the new canonical entry points

2. Date/time
   2026-03-15 13:05:00 +08:00

3. Hypothesis
   After the restructure, the canonical tutorial entry points should still be
   able to regenerate the expected Murat and driving-dataset inputs, and the
   validation flow should run against the editable local `pyblinker` checkout.

4. Files inspected
- `tutorial/murat_sequence/step1_prepare_dataset.py`
- `tutorial/murat_sequence/step2_run_blinker.py`
- `tutorial/raja_sequence/step3_run_blinker.py`
- `src/validation/fresh_compare_from_csv.py`
- `src/validation/fresh_compare_subjects.py`
- `src/validation/run_murat_full_with_status.py`
- `config/config.yaml`
- local `pyblinker` clone under `pyblinker/`

5. Files changed
- `docs/findings/025_post_restructure_input_regeneration_and_smoke_validation.md`

6. Exact change made
- Created this run log before the preparation and validation smoke run.
- Confirmed the editable `pyblinker` install resolves to the local clone at
  `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker`.
- Ran `tutorial/murat_sequence/step1_prepare_dataset.py --root D:\dataset\murat_2018 --verbose`
  to rebuild Murat FIF/EDF inputs.
- Ran `tutorial/murat_sequence/step2_run_blinker.py --root D:\dataset\murat_2018 --verbose`
  to verify the canonical MATLAB-backed Murat runner still works from the new
  repository layout. All 75 outputs were already present and were skipped.
- Ran `tutorial/raja_sequence/step3_run_blinker.py --root D:\dataset\drowsy_driving_raja_processed --verbose`
  to generate the missing Raja EDF and `blinker_results.pkl` outputs.
- Ran `tutorial/murat_sequence/step3_validate_pyblinker.py --prefix exp07smoke --selection top --n 2 --force-rerun`
  as a Murat smoke validation run from the new canonical entry point.
- Ran `tutorial/raja_sequence/step4_validate_pyblinker.py --prefix drvsmoke01 --subjects S1,S2 --force-rerun --restrict-py-to-comparison-channels`
  as a driving-dataset smoke validation run from the new canonical entry point.

7. Why the change was made
- The post-restructure environment no longer had Murat FIF/EDF inputs.
- The local editable `pyblinker` workflow needed to be confirmed before running validation.
- This run should leave a resumable record even if preparation or validation fails midway.

8. MATLAB reference used
- EEGLAB root: `D:\code development\matlab_plugin\eeglab2025.1.0`
- Blinker plugin: `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Blinker1.2.0`

9. Validation scope
- Confirm local editable `pyblinker` import path
- Regenerate Murat FIF/EDF inputs
- Regenerate Murat MATLAB Blinker outputs
- Regenerate driving-dataset MATLAB Blinker outputs as needed
- Run canonical validation entry points on focused smoke scopes

10. Before/after metrics
- Before:
  - `murat_2018` had `75` `.mat` files but `0` `.fif` and `0` `.edf`
  - `driving_dataset` had `.fif` inputs and partial `.edf` coverage
  - Python imported `pyblinker` from `site-packages`, not the local clone
- After:
  - `murat_2018` now has `75` `.fif`, `75` `.edf`, and `75` `blinker_results.pkl`
  - `driving_dataset` now has `22` `blinker_results.pkl` outputs across the
    available subject folders
  - Python resolves `pyblinker` from
    `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker`
  - Murat smoke scope:
    - `9636595`: `share_within_tolerance_percent = 100.0`
    - `9636571`: `share_within_tolerance_percent = 100.0`
    - aggregate reports:
      - `reports/validation/exp07smoke_top2_summary.csv`
      - `reports/validation/exp07smoke_top2_overall.json`
  - driving-dataset smoke scope:
    - `S1`: `share_within_tolerance_percent = 100.0`
    - `S2`: `share_within_tolerance_percent = 100.0`
    - aggregate reports:
      - `reports/validation/drvsmoke01_driving_dataset_2subjects_summary.csv`
      - `reports/validation/drvsmoke01_driving_dataset_2subjects_overall.json`

11. Whether the change was kept or reverted
- Kept

12. Next recommended step
- Scale the same canonical entry points from smoke scope to the full Murat and
  driving-dataset validation sets.
- If any PyBlinker logic changes are needed during broader validation, bump the
  experiment prefix and re-run both datasets according to `good_practice.md`.
