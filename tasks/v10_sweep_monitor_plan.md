# V10 Parameter Sweep — Launch & Autonomous Monitoring Plan

**Purpose**: Self-contained execution plan for a new Claude context to launch the V10
sweep, monitor it via `ScheduleWakeup`, and autonomously fix + resume on errors.

**Pre-reading required** (read before any phase):
- `CLAUDE.md` — project constraints, folder layout, reviewing agent protocol
- `lessons.md` — known failure patterns (esp. V8 int/float crash, V9 OOM)
- `tasks/sweep_plan.md` — full parameter space spec for the sweep
- `scripts/param_sweep.py` — sweep entry point (current state)
- `src/indicators/zscore_variants.py` — contains `_rolling_rank` (V9 OOM fix already applied)

---

## Phase 0: Pre-flight Verification (ALWAYS FIRST)

Before launching, verify all V9 fixes are in place and the sweep is ready.

### Checklist — run these greps/reads:

```bash
# 1. Confirm LOG_INTERVAL is 200 (not 100)
grep "LOG_INTERVAL" scripts/param_sweep.py
# Expected: LOG_INTERVAL      = 200

# 2. Confirm gc is imported
grep "^import gc" scripts/param_sweep.py
# Expected: import gc

# 3. Confirm del df_ind is present after generate_signals
grep -A2 "del df_ind" scripts/param_sweep.py
# Expected: del df_ind  # free ~17-col 2M-row DF before backtest allocates

# 4. Confirm gc.collect() every 50 combos
grep "gc.collect" scripts/param_sweep.py
# Expected: gc.collect()

# 5. Confirm _rolling_rank uses pandas (not as_strided)
grep "rolling.*rank\|as_strided" src/indicators/zscore_variants.py
# Expected: .rolling(window, ...).rank(...) — NO as_strided in _rolling_rank

# 6. Confirm output path for V10
# Parquet should go to: data/ml/ml_dataset_v10.parquet
# Log will be auto-named: sweep_run_v10.log
```

### Anti-patterns to reject:

- `LOG_INTERVAL = 100` → must be 200
- `as_strided` inside `_rolling_rank` → must be pandas rolling
- No `del df_ind` / `del df_sig` / `del results` in main loop → must be present
- No `gc.collect()` every 50 combos → must be present

**Do not proceed to Phase 1 if any check fails. Fix first.**

---

## Phase 1: Launch V10 Sweep in Background

### Command

```bash
python scripts/param_sweep.py \
  --range-mode v10 \
  --combinations 10000 \
  --output data/ml/ml_dataset_v10.parquet \
  --hours 300 \
  > sweep_run_v10.log 2>&1
```

> **Note**: The sweep script tees output to the log file automatically when
> `--output` is given. The redirect `> sweep_run_v10.log 2>&1` is a
> belt-and-suspenders fallback; the in-process `_Tee` class handles it.
> Use `run_in_background=True` in the Bash tool.

### Confirm launch

After spawning, immediately tail the log to verify the sweep started and
the cache-build phase prints without error:

```bash
tail -20 sweep_run_v10.log
```

Expected opening lines:
```
[sweep] Loading bars …
[sweep] Cache built in Xs — z_windows=N  vol_windows=N  ema_periods=N
[sweep] Starting 10000 combos …
```

If the log is empty or shows a Python traceback → **stop, fix, do not proceed**.

---

## Phase 2: Initial Monitor Setup (ScheduleWakeup — 120 s)

Immediately after confirming the sweep started, call `ScheduleWakeup` with
`delaySeconds=120` (2 min). This is the "before we know speed" interval.

### On each wakeup, run the monitor routine (Phase 3).

The monitor routine will compute the adaptive interval and re-schedule itself.

---

## Phase 3: Monitor Routine (runs every wakeup)

On each `ScheduleWakeup` firing, execute ALL of the following steps:

### Step 1 — Parse progress

```bash
tail -200 sweep_run_v10.log
```

Extract from the most recent `[sweep] N/10000 (P%) | Tm | last combo #ID` line:
- `n_done` — combos completed so far
- `elapsed_m` — elapsed minutes
- `last_combo_id` — last processed combo

### Step 2 — Count errors

```bash
grep -c "combo.*error:" sweep_run_v10.log
```

Also extract the unique error messages:
```bash
grep "combo.*error:" sweep_run_v10.log | grep -oP "error: .*" | sort | uniq -c | sort -rn | head -10
```

### Step 3 — Compute combo rate and adapt interval

```
rate = n_done / elapsed_m   (combos per minute)
next_delay_seconds = ceil(200 / rate * 60)
next_delay_seconds = clamp(next_delay_seconds, 60, 3600)
```

Example: at V9 speed (~36 combos/min), 200 combos ≈ 5.5 min → 330 s.

### Step 4 — Report to user

Output a concise status block:
```
[V10 Monitor] n_done/10000 (P%) | Xm elapsed | ~Y min remaining
Errors so far: N  (top: "...message...")
Combo rate: ~Z /min | Next wakeup in Xs (~200 combos)
```

### Step 5 — Check for completion or fatal state

- If log contains `[sweep] Done` → proceed to Phase 5 (post-completion). Do NOT re-schedule.
- If error rate > 30% of completed combos → trigger Phase 4 (error handling).
- If log contains `[sweep] FATAL` → trigger Phase 4 immediately.

### Step 6 — Re-schedule

Call `ScheduleWakeup` with `delaySeconds = next_delay_seconds` computed in Step 3.
Pass the same `/loop`-style prompt so the next wakeup re-enters this routine.

---

## Phase 4: Autonomous Error Handling

**Applies when**: new error type appears that wasn't present in V9, OR error rate
exceeds 30%, OR a `FATAL` is logged.

**Permissions**: with `--dangerously-skip-permissions`, proceed without user approval.
Still post a message to the user after fixing, describing exactly what changed.

### Step 1 — Pause the sweep

```bash
echo "" > stop_sweep.txt
```

The sweep checks `stop_sweep.txt` between combos and halts cleanly, flushing the
manifest. Wait one monitor cycle (or tail the log) to confirm the halt line:
```
[sweep] stop_sweep.txt detected — stopping.
```

### Step 2 — Diagnose the error

Read the full traceback from the log:
```bash
grep -A 40 "combo.*error:" sweep_run_v10.log | tail -80
```

Identify:
- **Error type**: OOM (`ArrayMemoryError`), type mismatch (`ArrowTypeError`),
  key error, import error, etc.
- **Root cause file and line number**
- **Whether it's a known pattern** (check `lessons.md`)

### Step 3 — Fix

Apply the minimal fix to the relevant file. Common patterns:

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `ArrayMemoryError: (N, window) float64` | New rolling fn uses `as_strided` | Replace with pandas rolling |
| `ArrowTypeError: int64 vs double` | New branch uses `int()` for nullable col | Change to `float()` |
| `KeyError: 'col_name'` | New column missing in combo dict | Add to `_sample_combos` and `_COMBO_META_KEYS` |
| `ArrayMemoryError` on small shape | Heap fragmentation | Reduce GC interval from 50 → 25 |

### Step 4 — Update lessons.md

Add an entry following the template in that file before resuming.

### Step 5 — Resume

```bash
rm stop_sweep.txt
```

Re-launch with the **same command** as Phase 1 (same `--output` path).
The sweep reads the manifest, detects already-completed `combo_id`s, and skips them.

Confirm resume by tailing the log:
```bash
tail -5 sweep_run_v10.log
```

Expected: `[sweep] Skipping N already-completed combo IDs.` followed by progress lines.

### Step 6 — Re-enter monitoring

Call `ScheduleWakeup` at 120 s (reset to initial interval since we lost timing state).
Re-enter Phase 3 on next wakeup.

---

## Phase 5: Post-Completion Verification

Triggered when log contains `[sweep] Done — 10000 combos`.

### Step 1 — Summary stats

```bash
tail -5 sweep_run_v10.log
grep -c "combo.*error:" sweep_run_v10.log
python -c "
import pandas as pd
df = pd.read_parquet('data/ml/ml_dataset_v10.parquet')
print('rows:', len(df))
print('combos:', df['combo_id'].nunique())
print('cols:', df.shape[1])
print(df.dtypes)
"
```

### Step 2 — Manifest check

```bash
python -c "
import json
m = json.load(open('data/ml/sweep_manifest_v10.json'))
completed = sum(1 for e in m if e['status'] == 'completed')
errors    = sum(1 for e in m if e['status'].startswith('error'))
print(f'completed: {completed}  errors: {errors}  total: {len(m)}')
"
```

### Step 3 — Schema sanity

Confirm no mixed-type columns (would break ML pipeline):
```bash
python -c "
import pyarrow.parquet as pq
schema = pq.read_schema('data/ml/ml_dataset_v10.parquet')
print(schema)
"
```

### Step 4 — Report to user

Post a final status message:
```
V10 sweep complete.
Total combos: 10000 | Completed: N | Errors: M (X%)
Parquet rows: ~R | Columns: C
Schema: OK / [issues if any]
Runtime: Xm
```

If errors > 5%: note which error types dominated and whether a fix is warranted
before ML training.

---

## Key Constants (V10 context)

| Item | Value |
|------|-------|
| Output Parquet | `data/ml/ml_dataset_v10.parquet` |
| Log file | `sweep_run_v10.log` |
| Manifest | `data/ml/sweep_manifest_v10.json` (auto-named by sweep) |
| Stop trigger | `stop_sweep.txt` in repo root |
| LOG_INTERVAL | 200 combos |
| BATCH_SIZE | 20 combos |
| GC interval | 50 combos |
| Initial monitor delay | 120 s |
| Adapted monitor delay | `ceil(200 / rate_combos_per_min * 60)` s |
| Error-rate threshold for pause | > 30% of completed combos |

---

## Permissions Note

With `--dangerously-skip-permissions` active, Claude will:
- Fix errors and resume **without waiting for user approval**
- **Always post a message** after any autonomous fix describing what changed
- **Never** push to remote, delete branches, or drop the Parquet dataset without explicit user instruction
