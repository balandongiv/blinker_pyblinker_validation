# Agent Runbook For Long Validation Experiments

1. Title
   Good practice runbook for long PyBlinker validation experiments with live
   status, logs, experiment versioning, and investigation notes.

2. Date/time
   2026-03-15 12:30:00 +08:00

3. Purpose
   This document captures the working pattern used for long `murat_2018` and
   `driving_dataset` validation runs so the same approach can be reused later.

4. Core principles

## 4.1 Make every long run reproducible
- Always use a clear experiment prefix such as `exp06`, `drvexp05`, or `exp07`.
- Keep output filenames deterministic, for example:
  - `<prefix>_pyblinker_results.pkl`
  - `<prefix>_top74_summary.csv`
  - `<prefix>_top74_overall.json`
- If shared logic changes, use a new experiment prefix.
- Do not mix artifacts from different logic versions under one prefix.

## 4.2 Separate smoke runs from full sweeps
- Start with a small scope first:
  - top 2
  - top 10
  - one problematic subject
- Only scale to the full dataset after the smoke scope is clean.
- After a reset, confirm a small scope before the full sweep.

## 4.3 Always leave a paper trail
- Create a markdown note before or alongside each meaningful run or fix.
- Record:
  - why the run is happening
  - experiment prefix
  - dataset
  - subject scope
  - files changed
  - commands used
  - before and after metrics
  - whether the change was kept
- Store these notes under `docs/findings/`.

## 4.4 Treat observability as part of the experiment
- For long jobs, do not rely on a silent terminal.
- Write:
  - a rolling text log
  - a live status JSON
  - a live status Markdown summary
- Update those files on a heartbeat and after each completed recording.

## 4.5 Verify outcomes, not just process launch
- After starting a long background job, immediately verify:
  - the process exists
  - the log file exists
  - the status file exists
  - the completed count increases after a short wait

## 4.6 Keep validation logic and run orchestration separate
- Comparison logic belongs in reusable modules under `src/validation/`.
- Long-run orchestration belongs in a dedicated runner such as
  `src/validation/run_murat_full_with_status.py`.
- The runner should not duplicate detector logic.

## 4.7 Make interruptions survivable
- Write progress to disk continuously.
- Prefer per-subject output files so finished work is not lost.
- Prefer a background process for multi-hour jobs.
- Keep run state in files that can be opened while the process is still running.

5. Repository locations

- Canonical validation code: `src/validation/`
- Aggregate experiment outputs: `reports/validation/`
- Investigation notes: `docs/findings/`
- Tutorial entry points: `tutorial/`

6. Concrete pattern used in this project

## 6.1 Validation runner
For long Murat reruns, use:

- [run_murat_full_with_status.py](/c:/Users/balan/IdeaProjects/blinker_pyblinker_validation/src/validation/run_murat_full_with_status.py)

This runner:
- force-reruns an ordered Murat sweep
- writes per-subject outputs into dataset folders
- writes a rolling log
- writes live status JSON and Markdown files
- updates status every heartbeat and after each completed recording

## 6.2 Live files
Live files are written under:

- [reports/validation](/c:/Users/balan/IdeaProjects/blinker_pyblinker_validation/reports/validation)

Typical examples:
- `exp06_top74_live_status.json`
- `exp06_top74_live_status.md`
- `exp06_top74_live_log.txt`

Final experiment artifacts are written to the same folder, for example:
- `exp06_top74_summary.csv`
- `exp06_top74_overall.json`

## 6.3 Markdown investigation trail
Keep one markdown file per meaningful investigation or run under:

- [docs/findings](/c:/Users/balan/IdeaProjects/blinker_pyblinker_validation/docs/findings)

7. How to reproduce the live-status approach manually

## 7.1 Start the background run
Use PowerShell `Start-Process` so the run is detached from the interactive terminal:

```powershell
Start-Process `
  -FilePath "python" `
  -ArgumentList "-m","src.validation.run_murat_full_with_status","--prefix","exp06","--selection","top","--n","74","--force-rerun" `
  -WorkingDirectory "C:\Users\balan\IdeaProjects\blinker_pyblinker_validation"
```

## 7.2 Watch live Markdown status
```powershell
Get-Content C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\reports\validation\exp06_top74_live_status.md -Wait
```

## 7.3 Watch the rolling log
```powershell
Get-Content C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\reports\validation\exp06_top74_live_log.txt -Tail 30 -Wait
```

## 7.4 Check the background process
```powershell
Get-Process python
```

8. What to ask an agent in the future

Use a prompt with these elements:

```text
Run a fresh full validation sweep for <dataset> using experiment prefix <expNN>.

Requirements:
1. Do not reuse old experiment prefixes after logic changes.
2. Create a markdown investigation log before the run.
3. For long runs, create or reuse a runner that writes:
   - a rolling text log
   - a live status JSON
   - a live status Markdown file
4. Start the run in the background when appropriate and verify it is actually progressing.
5. Tell me exactly which files I can watch live.
6. When the run finishes, report the final summary CSV and overall JSON paths and the key metrics.
7. If any logic changes are made, rerun the previously validated scopes under a new experiment prefix.
```

If you want the same documentation discipline, add:

```text
Document every meaningful investigation or fix attempt in a markdown file under docs/findings.
```

9. Good practice checklist

## 9.1 Before running
- Confirm the dataset path and subject list.
- Confirm the experiment prefix.
- Decide whether this is:
  - smoke scope
  - staged batch
  - full sweep
- Create the markdown note first.
- Decide whether to reuse or force-rerun existing outputs.

## 9.2 During the run
- Keep a live status file.
- Keep a rolling log.
- Verify the process is alive.
- Verify the completed count is increasing.
- If there is a long silence, inspect the log and process state before restarting.

## 9.3 After the run
- Check:
  - summary CSV exists
  - overall JSON exists
  - expected per-subject pickle files exist
- Confirm final metrics, not only that files were written.
- Record the results in a markdown note.

## 9.4 If logic changes
- Increment the experiment prefix.
- Ignore or archive old experiment artifacts for the new logic version.
- Rerun previously validated groups to ensure no regression.
- Update the markdown trail with:
  - what changed
  - why it changed
  - what got revalidated

10. Recommended default workflow
1. Create a markdown investigation note.
2. Choose a fresh experiment prefix.
3. Run a smoke scope first.
4. If clean, start the longer run with live status output.
5. Watch the live status Markdown and rolling log.
6. Verify final metrics from summary CSV and overall JSON.
7. Write a final markdown note with outcomes and next steps.
