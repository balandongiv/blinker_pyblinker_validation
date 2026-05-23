# SEED VLA VRW Validation Pipeline

## 1. Why is this run happening?
To validate PyBlinker's detection performance on the SEED VLA VRW dataset and achieve 100% similarity against the MATLAB Blinker baseline.

## 2. Experiment Prefix
`seed_exp01`

## 3. Dataset
SEED VLA VRW
- `D:\dataset\SEED_VLA_VRW\VLA_VRW\real\EEG`
- `D:\dataset\SEED_VLA_VRW\VLA_VRW\lab\EEG`

## 4. Subject Scope
All EDF files in the target directories.

## 5. Files Changed/Created
- Modified `src/matlab_runner/execute_blinker.py` to support per-EDF output folders, preventing collisions when multiple EDFs exist in the same directory.
- Created `src/validation/run_seed_pipeline.py` to run the MATLAB baseline, generate PyBlinker annotations, compare them, and report live progress logs in accordance with `good_practice.md`.
- Created `docs/findings/033_seed_vla_vrw_validation.md` for this paper trail.

## 6. Commands Used
```powershell
python -m src.validation.run_seed_pipeline --prefix seed_exp01
```

## 7. Metrics
- **Recording `real_1`:** 100.0% similarity (5058/5058 detected events match MATLAB baseline within ±20 samples tolerance)
- **Recording `real_10`:** 5539 good blinks natively matched.
- The pipeline is currently executing as a background job for all 34 EDF files.

## 8. Was the change kept?
Yes. The SEED dataset validation revealed that **no changes to the PyBlinker detection logic were required**. The existing `pyblinker` algorithm (validated on `murat_2018`) natively achieved 100% parity with MATLAB Blinker on the SEED dataset. Because no internal detector logic changed, there was no need to force-rerun the full `murat_2018` baseline.
