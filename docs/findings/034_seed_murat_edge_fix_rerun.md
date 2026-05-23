# SEED And Murat Edge-Fix Rerun

## 1. Why is this run happening?
To fix the remaining SEED edge mismatches documented in `issue.md`, then rerun both the SEED VLA VRW validation and the full `murat_2018` validation under fresh experiment prefixes to confirm final 100% parity still holds.

## 2. Experiment Prefixes
- Final kept runs:
  - `seed_exp04`
  - `exp08`
- Discarded intermediate runs:
  - `seed_exp02`
  - `exp07`
  - `seed_exp03`

## 3. Datasets
- SEED VLA VRW
  - `D:\dataset\SEED_VLA_VRW\VLA_VRW\real\EEG`
  - `D:\dataset\SEED_VLA_VRW\VLA_VRW\lab\EEG`
- `murat_2018`
  - `D:\dataset\murat_2018`

## 4. Subject Scope
- SEED: all EDF files in the lab and real folders
- Murat: ordered top 74 full sweep

## 5. Files Changed
- `pyblinker/pyblinker/blinker/stroke_utils.py`
- `pyblinker/pyblinker/pipeline_steps.py`
- `pyblinker/pyblinker/utils/statistics_utils.py`
- `pyblinker/test/blinker_pyblinker_comparison/test_h_edge_case_regressions.py`

## 6. Commands Used
```powershell
python -m pytest -q pyblinker/test/blinker_pyblinker_comparison/test_h_edge_case_regressions.py pyblinker/test/blinker_pyblinker_comparison/test_f_fitblink_terminal_edge_case.py pyblinker/test/blinker_pyblinker_comparison/test_a2_stat.py
python -m src.validation.run_seed_pipeline --prefix seed_exp04 --force-rerun
python -m src.validation.run_murat_full_with_status --prefix exp08 --selection top --n 74 --force-rerun --max-workers 2
```

## 7. Before/After Metrics
- Before fix:
  - `seed_exp01` had `lab_8 = 99.9500249875%`
  - `seed_exp01` had `real_3 = 99.9714856002%`
- Smoke recheck after fix:
  - `lab_8 = 100.0%`
  - `real_3 = 100.0%`
- Murat targeted regression recheck after narrowing the fix back to SEED:
  - `9636622 = 100.0%`
  - `9636592 = 100.0%`
- Full-run results:
  - `seed_exp04`: all 34/34 recordings at `100.0%`
  - `exp08`: all 74/74 recordings at `100.0%`

## 8. Was the change kept?
Yes. The final kept change uses a boundary-only SEED amplitude-gate rescue plus the terminal-edge cleanup and was validated by the completed `seed_exp04` and `exp08` reruns.
