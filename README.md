# blinker_pyblinker_validation

`blinker_pyblinker_validation` is a research validation harness for comparing
legacy MATLAB `Blinker` outputs against `pyblinker` on real public datasets.

This repository is meant to be used alongside an editable `pyblinker` checkout.
The intended workflow is:

1. prepare canonical MATLAB-backed inputs
2. run MATLAB `Blinker`
3. run fresh `pyblinker`
4. compare blink-region outputs
5. fix `pyblinker` immediately if validation exposes a bug
6. rerun validation with a new experiment version

The repository currently supports two datasets:

- `murat_2018`
- `driving_dataset`

## What To Run

If you want the full end-to-end reproducibility pipeline, run:

```powershell
python run_full_repro_pipeline.py
```

That single script is the public-facing full-run entry point. It:

1. prepares Murat FIF and EDF files when needed
2. runs MATLAB `Blinker` for Murat
3. runs MATLAB `Blinker` for the driving dataset
4. runs the full PyBlinker-vs-Blinker sweep for Murat
5. runs the full PyBlinker-vs-Blinker sweep for the driving dataset
6. writes a combined pipeline log and manifest under `reports/validation`

If the repository is correctly configured and the datasets are present, that
single command is the closest thing to a one-click academic reproduction path.

## Repository Purpose

This repository does not replace `pyblinker`. It exists to validate `pyblinker`
against the legacy MATLAB implementation under realistic dataset conditions.

The goal is not only to raise metrics, but to identify the first point of
divergence when MATLAB and Python disagree and to preserve an auditable record
of each meaningful experiment.

## Relationship To `pyblinker`

`pyblinker` should remain installed in editable mode.

Typical setup:

- validation harness: `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation`
- editable `pyblinker`: `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker`

Recommended install:

```powershell
pip install -e .
pip install -e .\pyblinker
```

Why editable mode matters:

- validation can expose a bug in `pyblinker`
- the `pyblinker` code can be patched directly
- the same validation commands can be rerun immediately
- no reinstall cycle is required

When shared logic changes, follow `good_practice.md` and use a new experiment
version or run id.

## Repository Layout

```text
blinker_pyblinker_validation/
├── config/                       Configuration and dataset path defaults
├── docs/
│   └── findings/                Investigation logs and experiment history
├── mock_data/                   Small repository-local fixtures
├── reports/
│   └── validation/              Aggregate results, pipeline manifests, live logs
├── run_full_repro_pipeline.py   Clickable top-level full reproduction entry point
├── src/
│   ├── matlab_runner/           MATLAB / EEGLAB orchestration helpers
│   ├── murat/                   Murat data preparation helpers
│   ├── ui_murat/                Murat UI code
│   ├── ui_raja/                 Raja UI code
│   ├── utils/                   Shared utilities used by tutorial flows
│   └── validation/              Canonical validation and comparison logic
├── tests/                       Repository-local tests
├── tutorial/
│   ├── murat_sequence/          Tutorial and staged Murat entry points
│   ├── raja_sequence/           Tutorial and staged Raja entry points
│   └── *.py                     UI and demo wrappers
├── good_practice.md             Runbook for long validation experiments
├── pyproject.toml
└── README.md
```

Core validation code lives under `src/validation`. Tutorial and reproducibility
entry points live under `tutorial`.

## Prerequisites

### Python environment

Use the environment that contains:

- `mne`
- `numpy`
- `pandas`
- MATLAB Engine for Python
- the dependencies required by `pyblinker`

### MATLAB / EEGLAB

Expected MATLAB reference installation:

- EEGLAB root: `D:\code development\matlab_plugin\eeglab2025.1.0`
- Blinker plugin: `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Blinker1.2.0`

Default paths are configured in `config/config.yaml`.

### Dataset locations

Expected default dataset roots:

- `murat_2018`: `D:\dataset\murat_2018`
- `driving_dataset`: `D:\dataset\drowsy_driving_raja_processed`

These can be overridden through CLI arguments when needed.

## End-To-End Reproduction

### Fast path

Run the full pipeline with default locations:

```powershell
python run_full_repro_pipeline.py
```

### Fresh validation rerun

If you want to overwrite existing MATLAB outputs and fresh PyBlinker outputs:

```powershell
python run_full_repro_pipeline.py --run-id full_repro_v2 --force-murat-prepare --force-matlab --force-validation
```

Important rule:

- if `pyblinker` logic changed, use a new `--run-id`

The pipeline will derive:

- Murat prefix: `<run-id>_murat`
- driving prefix: `<run-id>_driving`

### What the full pipeline executes

The top-level runner orchestrates these canonical steps:

1. `tutorial/murat_sequence/step1_prepare_dataset.py`
2. `tutorial/murat_sequence/step2_run_blinker.py`
3. `tutorial/raja_sequence/step3_run_blinker.py`
4. `tutorial/murat_sequence/step3_validate_pyblinker.py`
5. `tutorial/raja_sequence/step4_validate_pyblinker.py`

By default:

- Murat preparation skips download and converts existing `.mat` files
- MATLAB Blinker steps reuse existing outputs unless forced
- validation steps reuse only good prefixed PyBlinker outputs unless forced

## Canonical Entry Points

### One-click full pipeline

- `run_full_repro_pipeline.py`
- `src/validation/run_full_repro_pipeline.py`

Use this when you want a full publishable reproduction run and a combined
manifest.

### Murat validation

- `tutorial/murat_sequence/step3_validate_pyblinker.py`
- `src/validation/run_murat_full_with_status.py`

Example:

```powershell
python tutorial\murat_sequence\step3_validate_pyblinker.py --prefix exp06 --selection top --n 74 --force-rerun
```

### Driving-dataset validation

- `tutorial/raja_sequence/step4_validate_pyblinker.py`
- `src/validation/fresh_compare_subjects.py`

Example:

```powershell
python tutorial\raja_sequence\step4_validate_pyblinker.py --prefix drvexp05 --subjects S1,S2,S3,S4,S5,S6,S7,S10,S11,S12,S13,S16,S17,S18,S19,S20,S21,S22,S23,S24,S26,S27 --restrict-py-to-comparison-channels --continue-on-failure --force-rerun
```

### Focused Murat subsets

- `src/validation/fresh_compare_from_csv.py`

Example:

```powershell
python -m src.validation.fresh_compare_from_csv --selection top --n 10 --prefix exp07 --force-rerun
```

## Outputs

### Per-recording PyBlinker outputs

Fresh PyBlinker outputs are written beside each source recording using the
selected prefix.

Examples:

- `D:\dataset\murat_2018\9636595\full_repro_v1_murat_pyblinker_results.pkl`
- `D:\dataset\drowsy_driving_raja_processed\S1\blinker_pyblinker_validation\full_repro_v1_driving_pyblinker_results.pkl`

### Aggregate reports

Aggregate outputs are written under:

- `reports/validation/`

Typical files from the full pipeline:

- `<run-id>_pipeline.log`
- `<run-id>_pipeline_manifest.json`
- `<run-id>_pipeline_manifest.md`
- `<run-id>_murat_top74_summary.csv`
- `<run-id>_murat_top74_overall.json`
- `<run-id>_driving_driving_dataset_22subjects_summary.csv`
- `<run-id>_driving_driving_dataset_22subjects_overall.json`

### Live Murat run visibility

The Murat full runner also writes:

- `<prefix>_top74_live_log.txt`
- `<prefix>_top74_live_status.json`
- `<prefix>_top74_live_status.md`

These are useful for long runs and power-outage recovery.

## How To Compare Against Academic Results

Use the following files after a full run:

1. `reports/validation/<run-id>_pipeline_manifest.md`
2. `reports/validation/<run-id>_murat_top74_summary.csv`
3. `reports/validation/<run-id>_murat_top74_overall.json`
4. `reports/validation/<run-id>_driving_driving_dataset_22subjects_summary.csv`
5. `reports/validation/<run-id>_driving_driving_dataset_22subjects_overall.json`

The combined manifest is the fastest way to confirm:

- which steps ran
- which prefixes were used
- where the final summary files are
- whether all recordings reached `100.0` share within tolerance

## Legacy Scripts

The old Murat comparison script
`tutorial/murat_sequence/legacy/step4_compare_pyblinker_vs_blinker_legacy.py`
is retained only as archival reference.

It is not part of the canonical pipeline anymore because it predates:

- fresh experiment-prefix runs
- live-status logging
- the new full reproducibility flow
- the current shared comparison logic in `src/validation`

## Good Practice

Use `good_practice.md` for long runs, bugfix investigations, and any logic
change that could affect one or both datasets.

Important rules:

- create a markdown finding log for each meaningful run or fix
- use a new experiment version after shared logic changes
- keep live logs and status files for long runs
- revalidate previously clean scopes when shared logic changes

## Troubleshooting

### The full pipeline cannot find `pyblinker`

Check that the editable install points at the intended checkout:

```powershell
python -c "import importlib.util; print(importlib.util.find_spec('pyblinker').submodule_search_locations)"
```

### MATLAB or EEGLAB is not found

Check:

- `config/config.yaml`
- the EEGLAB path on disk
- MATLAB Engine availability in the active environment

### Old outputs are being reused

Use a new `--run-id` after logic changes, or use:

```powershell
python run_full_repro_pipeline.py --run-id full_repro_v2 --force-validation
```

### A long Murat run is interrupted

Check:

- `reports/validation/<prefix>_top74_live_status.md`
- `reports/validation/<prefix>_top74_live_log.txt`

The runbook in `good_practice.md` documents the expected recovery pattern.

## Testing

Run repository-local tests with:

```powershell
pytest tests -q
```

If a change affects shared validation logic or editable `pyblinker` logic, rerun
the relevant Murat and driving-dataset validation scopes and record the result in
`docs/findings/`.
