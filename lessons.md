## Purpose

Persistent log of lessons learned and mistakes encountered while building/running this repo, so Claude Code (and humans) do not repeat them.

## How to use

- Before running new tooling commands (installs, notebook execution, backtest scripts), scan recent entries.
- When something fails, add a new entry immediately with the fix and a prevention rule.

## Entry template

### YYYY-MM-DD Short_title

- **What I ran**:
- **What happened (error summary)**:
- **Root cause**:
- **Fix**:
- **Prevention rule**:
- **Related files/commands**:

---

### 2026-04-10 nbconvert_cwd_flag_not_recognized

- **What I ran**: `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.cwd=<path> notebooks/01_backtest_and_log.ipynb`
- **What happened**: `ModuleNotFoundError: No module named 'src'` — kernel launched from wrong cwd; `sys.path.insert(0, ".")` adds `.` relative to the kernel's cwd, not the repo root.
- **Root cause**: `--ExecutePreprocessor.cwd` is not a recognized config option in nbconvert 7.x / nbclient 0.10.x. The flag is silently ignored.
- **Fix**: Use `nbclient.NotebookClient` directly in a Python one-liner with the `cwd` kwarg:
  ```python
  import nbformat
  from nbclient import NotebookClient
  with open('notebooks/01_backtest_and_log.ipynb') as f:
      nb = nbformat.read(f, as_version=4)
  NotebookClient(nb, timeout=600, kernel_name='python3', cwd='<repo_root>').execute()
  with open('notebooks/01_backtest_and_log.ipynb', 'w') as f:
      nbformat.write(nb, f)
  ```
- **Prevention rule**: Never use `--ExecutePreprocessor.cwd` with nbconvert. Always use `nbclient.NotebookClient(cwd=...)` for in-place notebook execution with a specific working directory.
- **Related files/commands**: `notebooks/01_backtest_and_log.ipynb`, `notebooks/02_plotly_exploration.ipynb`

