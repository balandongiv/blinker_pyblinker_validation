# 016 Driving FIF EDF Parity Alignment

1. Title
Align the driving-dataset FIF validation path to the MATLAB EDF channel grid and padded record length

2. Date/time
2026-03-14 15:14:33 +08:00

3. Hypothesis
The new driving-dataset FIF workflow is failing on subjects like `S11` not because of blink-stage logic, but because the `.fif` signal differs slightly from the MATLAB-side `.edf` signal. If the runner trims the FIF to the EDF channel set, quantizes those channels onto the EDF calibration grid, and matches the EDF tail padding, the comparison should recover exact MATLAB parity without changing detector logic.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\drvexp03_driving_dataset_1subjects_summary.csv`
- `D:\dataset\drowsy_driving_raja_processed\S11\S11.fif`
- `D:\dataset\drowsy_driving_raja_processed\S11\blinker_pyblinker_validation\S11.edf`
- `D:\dataset\drowsy_driving_raja_processed\S10\S10.fif`
- `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\S10.edf`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\016_driving_fif_edf_parity_alignment.md`

6. Exact change made
Extended `fresh_compare_subjects.py` so that `--restrict-py-to-comparison-channels` now:
- restricts the PyBlinker raw to the EDF channel intersection in EDF order
- when the Py raw is `.fif` and the comparison raw is `.edf`, quantizes each shared channel onto the EDF calibration grid using the EDF `cal`, `offsets`, and `units`
- pads or crops the FIF tail to match the EDF sample count, using the EDF-style repeated last-sample tail when padding is required
- updates the CLI help text so this parity behavior is explicit

7. Why the change was made
`S11` failed at `99.7815401419989` even after simple channel restriction. The direct raw probe showed:
- FIF and EDF channels fully overlap
- the FIF/EDF waveform mismatch on `eog_vert_right` was quantization-shaped
- the EDF had an extra repeated-sample tail that the FIF did not
Those are validation-input parity issues, so the smallest correct fix was to align the FIF input to the EDF representation before running the unchanged detector.

8. MATLAB reference used
- MATLAB comparison target: `D:\dataset\drowsy_driving_raja_processed\S11\blinker_pyblinker_validation\blinker_results.pkl`
- MATLAB-side raw path: `D:\dataset\drowsy_driving_raja_processed\S11\blinker_pyblinker_validation\S11.edf`

9. Validation scope
- Which subjects:
  - `S11` first as the gate subject
  - then `S11, S12, S13, S16, S17, S18, S19, S20, S21, S22, S23, S24, S26, S27`
- How many subjects:
  - `1` gate subject
  - `14` full requested batch before `S10`
- Which tests:
  - `C:\Users\balan\anaconda3\envs\pyblinker\python.exe -m pytest test/blinker_pyblinker_comparison -q`
  - `C:\Users\balan\anaconda3\envs\pyblinker\python.exe -m blinker_pyblinker_validation.fresh_compare_subjects --dataset driving_dataset --prefix drvexp04 --subjects S11 --py-raw-template '{id}/{id}.fif' --comparison-raw-template '{id}/blinker_pyblinker_validation/{id}.edf' --restrict-py-to-comparison-channels`
  - `C:\Users\balan\anaconda3\envs\pyblinker\python.exe -m blinker_pyblinker_validation.fresh_compare_subjects --dataset driving_dataset --prefix drvexp04 --subjects S11,S12,S13,S16,S17,S18,S19,S20,S21,S22,S23,S24,S26,S27 --py-raw-template '{id}/{id}.fif' --comparison-raw-template '{id}/blinker_pyblinker_validation/{id}.edf' --restrict-py-to-comparison-channels`

10. Before/after metrics
Before the parity fix:
- `S11` under `drvexp03` with restricted FIF channels still failed at `99.7815401419989`
- `total_detected = 1832`
- `total_ground_truth = 1830`
- `detected_only = 5`
- `ground_truth_only = 3`

After the parity fix:
- `S11` under `drvexp04` reached `100.0`
- requested batch `S11,S12,S13,S16,S17,S18,S19,S20,S21,S22,S23,S24,S26,S27` all reached `100.0`
- aggregate results in `drvexp04_driving_dataset_14subjects_overall.json`:
  - `total_detected_total = 32717`
  - `total_ground_truth_total = 32717`
  - `detected_only_total = 0`
  - `ground_truth_only_total = 0`
  - strict/lenient macro and micro `precision`, `recall`, `f1`, and `accuracy` all `1.0`
- focused tests remained `8 passed`

11. Whether the change was kept or reverted
Kept. The change is isolated to the validation runner and restored exact FIF-vs-EDF parity for the requested driving subjects without touching the detector pipeline or breaking the focused comparison tests.

12. Next recommended step
Proceed to `S10` under the same `drvexp04` FIF-parity path. If it still fails, treat it as the remaining special-case EDF-ingestion blocker and inspect the first stage where the Python result diverges from the MATLAB stored signal.
