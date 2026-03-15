# 023 Agent Runbook For Long Validation Experiments

1. Title
Good practice runbook for long pyblinker validation experiments with live status, logs, experiment versioning, and investigation notes

2. Date/time
2026-03-15 08:35:00 +08:00

3. Purpose
This document captures the working pattern used for the long `murat_2018` and `driving_dataset` validation runs so the same approach can be reused later, either manually or by asking an agent to do it.

4. Core principles

## 4.1 Make every long run reproducible
- Always use a clear experiment prefix such as `exp06`, `drvexp05`, `exp07`.
- Keep output filenames deterministic, for example:
  - `<prefix>_pyblinker_results.pkl`
  - `<prefix>_top74_summary.csv`
  - `<prefix>_top74_overall.json`
- If shared logic changes, use a new experiment prefix.
- Do not mix artifacts from different logic versions under one prefix.

## 4.2 Separate short proof runs from full sweeps
- Start with a small smoke scope first:
  - top 2
  - top 10
  - one problematic subject
- Only scale to the full dataset after the smoke scope is clean.
- For a long rerun after a reset, reuse the same validated logic and then scale to the full ordered sweep.

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
- This makes interrupted work resumable.

## 4.4 Treat observability as part of the experiment
- For long jobs, do not rely on a silent terminal.
- Write:
  - a rolling text log
  - a live status JSON
  - a live status Markdown summary
- Update those files on a heartbeat and after each completed subject.
- This makes it obvious whether the process is active, stuck, or finished.

## 4.5 Verify outcomes, not just process launch
- After starting a long background job, immediately verify:
  - process exists
  - log file exists
  - status file exists
  - completed count increases after a short wait
- Do not assume the run is healthy just because a process started.

## 4.6 Keep the validation logic and run orchestration separate
- The comparison logic belongs in reusable modules.
- The long-running orchestration belongs in a dedicated runner script.
- The runner should not duplicate detector logic.
- The runner should call the validated reusable functions and only add:
  - scheduling
  - status writing
  - logging
  - artifact naming

## 4.7 Make outages survivable
- Write progress to disk continuously.
- Prefer per-subject output files so finished work is not lost.
- Prefer a background process for multi-hour jobs.
- Keep the run state in files that can be opened from the IDE while the process is still running.

5. Concrete pattern used in this project

## 5.1 Validation runner
For the long full `murat_2018` rerun, a dedicated runner was added:

- [run_murat_exp06_full_with_status.py](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/run_murat_exp06_full_with_status.py)

This runner:
- force-reruns the ordered `exp06` murat sweep
- writes per-subject outputs into each subject folder
- writes a rolling log
- writes live status JSON and Markdown files
- updates status every heartbeat and after each completed recording

## 5.2 Live files
The live files written by the runner are:

- [exp06_top74_live_status.json](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/experiment_results/exp06_top74_live_status.json)
- [exp06_top74_live_status.md](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/experiment_results/exp06_top74_live_status.md)
- [exp06_top74_live_log.txt](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/experiment_results/exp06_top74_live_log.txt)

The final artifacts are:

- [exp06_top74_summary.csv](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/experiment_results/exp06_top74_summary.csv)
- [exp06_top74_overall.json](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/experiment_results/exp06_top74_overall.json)

## 5.3 Markdown investigation trail
The run history is kept under:

- [finding](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/finding)

Examples:
- [021_murat_exp06_full_ordered_sweep.md](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/finding/021_murat_exp06_full_ordered_sweep.md)
- [022_exp06_live_status_runner.md](/c:/Users/balan/IdeaProjects/pyblinker/blinker_pyblinker_validation/finding/022_exp06_live_status_runner.md)

6. How to reproduce the live-status approach manually

## 6.1 Start the background run
Use PowerShell `Start-Process` so the long run is detached from the current interactive terminal:

```powershell
Start-Process `
  -FilePath "C:\Users\balan\anaconda3\envs\pyblinker\python.exe" `
  -ArgumentList "-m","blinker_pyblinker_validation.run_murat_exp06_full_with_status" `
  -WorkingDirectory "C:\Users\balan\IdeaProjects\pyblinker"
```

## 6.2 Watch live Markdown status
```powershell
Get-Content C:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp06_top74_live_status.md -Wait
```

## 6.3 Watch the rolling log
```powershell
Get-Content C:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp06_top74_live_log.txt -Tail 30 -Wait
```

## 6.4 Check the background process
```powershell
Get-Process python
```

7. What to ask an agent in the future

Use a prompt with these elements:

```text
Run a fresh full validation sweep for <dataset> using experiment prefix <expNN>.

Requirements:
1. Do not reuse old experiment prefixes after logic changes.
2. Create a markdown investigation log before the run.
3. For long runs, create a dedicated runner that writes:
   - a rolling text log
   - a live status JSON
   - a live status Markdown file
4. Start the run in the background and verify it is actually progressing.
5. Tell me exactly which files I can watch live.
6. When the run finishes, report the final summary CSV and overall JSON paths and the key metrics.
7. If any logic changes are made, rerun the previously validated scopes under a new experiment prefix.
```

If you want the agent to preserve the same discipline used here, add:

```text
Document every meaningful investigation or fix attempt in a markdown file under blinker_pyblinker_validation/finding.
```

8. Good practice checklist

## 8.1 Before running
- Confirm the dataset path and subject list.
- Confirm the experiment prefix.
- Decide whether this is:
  - smoke scope
  - staged batch
  - full sweep
- Create the markdown note first.
- Decide whether existing outputs should be reused or force-rerun.

## 8.2 During the run
- Keep a live status file.
- Keep a rolling log.
- Verify the process is alive.
- Verify completed count is increasing.
- If there is a long silence, inspect the log and process state before restarting.

## 8.3 After the run
- Check:
  - summary CSV exists
  - overall JSON exists
  - expected per-subject pickle files exist
- Confirm final metrics, not only that files were written.
- Record the results in a markdown note.

## 8.4 If logic changes
- Increment the experiment prefix.
- Delete or ignore old experiment artifacts for the new logic version.
- Rerun previously validated groups to ensure no regression.
- Update the markdown trail with:
  - what changed
  - why it changed
  - what got revalidated

9. What made this approach useful
- The run could survive terminal interruptions.
- The status was visible without attaching to the process.
- The log proved real progress subject by subject.
- The investigation notes preserved reasoning, not just outputs.
- The experiment prefixes prevented silent mixing of old and new results.

10. Recommended default workflow for future long validations
1. Create a markdown investigation note.
2. Choose a fresh experiment prefix.
3. Run a smoke scope first.
4. If clean, create or reuse a dedicated full-run runner with live status output.
5. Start the full run in the background.
6. Watch the live status Markdown and rolling log.
7. Verify final metrics from summary CSV and overall JSON.
8. Write a final markdown note with outcomes and next steps.
