# 019 Test Prepare Event Tables Import Fix

1. Title
Repair the broken import path in `test_e_prepare_event_tables.py` before the fresh post-reset rerun

2. Date/time
2026-03-14 16:15:00 +08:00

3. Hypothesis
The focused comparison suite is currently failing because `test_e_prepare_event_tables.py` imports `prepare_event_tables` from an invalid module path. Fixing that import should restore the intended baseline without changing detector logic.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\test\blinker_pyblinker_comparison\test_e_prepare_event_tables.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\blink_compare.py`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\test\blinker_pyblinker_comparison\test_e_prepare_event_tables.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\019_test_prepare_event_tables_import_fix.md`

6. Exact change made
Replaced `from o.blink_compare import prepare_event_tables` with `from blinker_pyblinker_validation.blink_compare import prepare_event_tables`.

7. Why the change was made
The test could not import the target function, so the suite failed during collection and could not serve as a baseline for the rerun.

8. MATLAB reference used
None. This is a Python test harness repair only.

9. Validation scope
- Which subjects:
  - none
- How many subjects:
  - 0
- Which tests:
  - `C:\Users\balan\anaconda3\envs\pyblinker\python.exe -m pytest test/blinker_pyblinker_comparison -q`

10. Before/after metrics
Before:
- test collection failed with `ModuleNotFoundError: No module named 'o'`

After:
- `C:\Users\balan\anaconda3\envs\pyblinker\python.exe -m pytest test/blinker_pyblinker_comparison -q`
- `8 passed`

11. Whether the change was kept or reverted
Kept.

12. Next recommended step
Rerun the focused comparison suite, then proceed with the fresh dataset validations under new experiment prefixes.
