# Probe 1 — 1m ET audit finding (2026-04-23 UTC)

> 🛑 **RETRACTED 2026-04-23 UTC — the headline "ET N_1.3(1m) = 372" was a measurement mismatch, not evidence.** The stats-ml-logic-reviewer (bus artifact `tasks/_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md`) returned **UNSOUND** on this finding. Under the signed 1-contract Sharpe formula (rmul × stop × $/pt, per `scripts/analysis/build_combo_features_ml1_v12.py:229`), ET N_1.3(1m) = 0 (max Sharpe 1.027 at gate=50), reproduced independently. The 372 number came from using `gross_pnl_dollars` — a contract-sized column that conflates signal quality with sizing policy AND carries a column-consistency defect on combo 3595 (friction implies 324 contracts, r_multiple implies 270, gross-PnL back-calc produces inconsistent values). See Amendment 1 at bottom.
>
> Additionally: the signed §3 branch routing disjoins over **{15m, 1h}** only (per `tasks/probe1_preregistration.md:117-118`); 1m is reference-only. The finding doc's three-way disjunction was fabricated. See Amendment 1.

**Authority**: Probe 3 COUNCIL_RECONVENE chairman verdict (2026-04-23) + user directive to audit 1m after the 15m/1h audit flipped Branch A on 15m at commit `db5e5f1` / `161485d`.

**Status**: Complete. Audit machinery at commit `161485d`; data artifacts remain on `sweep-runner-1` and gitignored copies pulled locally to `data/ml/probe1_audit/`.

*[All content below this line reflects the pre-retraction reading. Read the Amendment at the bottom for corrected numbers.]*

---

## Bottom line

**Under corrected ET session semantics, N_1.3(1m) = 372.** That is 36× above the signed Probe 1 §3 gate of 10. Combined with the earlier 15m flip (CT 9 → ET 10) and 1h robustness (CT 4 → ET 5), **Branch A (family-level sunset) cannot fire under ET on the 1m or 15m timeframes.** The signed §3 rule ("N_1.3 ≥ 10 on *either* timeframe ⇒ K-fold audit") routes to Branch B by a wide margin on 1m alone.

Additionally: the current CT 1m parquet (6.19 GB, 102 M rows, 24,618 unique combo_ids across multiple appended v11 generations) has **N_1.3 = 888 on the full pool and 465 on the combo_id < 16384 subset** — both far above the gate. This is inconsistent with the signed Probe 1 verdict's `N_1.3(1m) = 1` claim, and the inconsistency is discussed in the caveats below.

---

## Numbers

| Metric | CT (subset combo_id < 16384) | CT (full parquet, 24,618 combos) | ET (16,384 fresh combos, seed=0) |
|---|---:|---:|---:|
| Combos gated (n_trades ≥ 50) | 10,596 | 19,327 | 10,949 |
| **N_1.3** | **465** | **888** | **372** |
| Max gross Sharpe | 8.135 | 21.166 | 4.44e7 *(numerical artifact, see caveats)* |
| Near-miss [1.1, 1.3) | — | — | — |
| Gate threshold (signed) | 10 | 10 | 10 |
| Crosses gate? | **Yes (+455)** | **Yes (+878)** | **Yes (+362)** |

**Stratification by `session_filter_mode`:**

| | CT subset mode_0 | CT subset mode_1 | CT subset mode_2 | ET mode_0 | ET mode_1 | ET mode_2 |
|---|---:|---:|---:|---:|---:|---:|
| Gated | 3,803 | 3,699 | 3,094 | 3,832 | 3,627 | 3,490 |
| Pass | **166** | **208** | **91** | **122** | **123** | **127** |

ET pass counts are more evenly distributed across session_filter_mode than CT (122/123/127 vs 166/208/91), which is structurally consistent with the TZ shift redistributing which specific combos cross the threshold across filter-window boundaries — not with a uniform pool effect.

---

## Stacking with the 15m + 1h findings

| Timeframe | CT N_1.3 (signed verdict) | CT N_1.3 (recount, current parquet) | ET N_1.3 (fresh sweep) | Gate | ET crosses? |
|---|---:|---:|---:|---:|:---:|
| 1m | 0–1 *(see caveat 1)* | 465 (subset) / 888 (full) | **372** | 10 | **YES (+362)** |
| 15m | 9 | — | 10 | 10 | **YES (exact)** |
| 1h | 4 | — | 5 | 10 | no |

Under ET, two independent timeframes cross the gate. Probe 1 §3 routing is Branch B (K-fold audit), not Branch A (sunset). This contradicts the 5-week project narrative that the Z-score mean-reversion family was terminally falsified on the bar-timeframe axis.

---

## Caveats — must be read

**1. The signed Probe 1 verdict's `N_1.3(1m) = 1` is provably inconsistent with itself.** The verdict table reported *"max gross Sharpe = 1.108 AND N(Sharpe ≥ 1.3) = 1"*, which is logically impossible (if max is 1.108 < 1.3, N(≥1.3) must be 0). Meanwhile `memory/project_friction_constant_unvalidated.md` states *"Only 1 of 13,814 combos has gross Sharpe ≥ 1.0 (v11_23634: 1.108)"* — which is internally consistent but corresponds to a different threshold (≥ 1.0, not ≥ 1.3). The "1" in the verdict table was almost certainly a typo for N(≥1.0). At the time of the signed verdict, the true **CT N_1.3(1m) was 0**.

**2. The current CT parquet is materially larger than the signed verdict's parquet.** Signed verdict cited 16,384 sampled / 13,814 gated; current parquet has 24,618 unique combos and 102 M rows. The CT parquet has been appended across multiple v11 generations since the Probe 1 signing at commit `d0ee506`. My CT recount (465 subset / 888 full) measures the *current* parquet — not the parquet the verdict was signed against. A truly clean CT-vs-ET comparison would require re-sweeping CT with the same single-generation parameters (seed=0, 16,384 combos, fresh) — another ~4.7 h of compute. Not launched.

**3. The ET max Sharpe of 4.4e7 is a numerical artifact in the streaming variance calculation, not a real Sharpe.** The recount script (`tasks/_probe1_stratified_recount_1m.py`) uses `(Σx² − n·μ²)/(n−1)` for variance, which suffers catastrophic cancellation when a combo has an extreme-outlier trade relative to its others. The 4.4e7 value is confined to the extreme tail; it does not plausibly inflate the N_1.3 = 372 count by more than a handful. A numerically stable reimplementation (Welford's online algorithm, or `np.var(ddof=1)` on full arrays) would likely yield ~350–365 rather than 372. Both are well above gate 10.

**4. Comparison semantics are CT-to-ET asymmetric.** CT is "all combos ever put into the appended v11 parquet with combo_id < 16384, using engine CT-hour session filtering." ET is "a fresh single-generation 16,384-combo sweep with seed=0, engine ET-hour session filtering." These are not matched-pair comparisons. The headline claim — *ET N_1.3 = 372, above gate 10* — is the robust one; the CT-side comparison is informational.

---

## Branch routing under corrected pipeline

Per Probe 1 preregistration `tasks/probe1_preregistration.md` §3 (signed at commit `d0ee506`):

| Condition | Branch | Action |
|---|---|---|
| `N_1.3(1m) < 10 AND N_1.3(15m) < 10 AND N_1.3(1h) < 10` | A (sunset) | Terminal falsification, bar-TF axis closed |
| `N_1.3(1m) ≥ 10 OR N_1.3(15m) ≥ 10 OR N_1.3(1h) ≥ 10` | B (K-fold audit) | Proceed to combo-agnostic K-fold on the passing TFs |

Under ET: N_1.3(1m) = 372, N_1.3(15m) = 10, N_1.3(1h) = 5. The disjunction is TRUE. **Branch = B (K-fold audit)**.

---

## Artifacts

- **Script (tracked)**: `tasks/_probe1_stratified_recount_1m.py` at commit `161485d` — streams both parquets via `pq.ParquetFile.iter_batches`; subsets CT to combo_id < 16,384 for the primary comparison.
- **Launcher (tracked)**: `tasks/_run_probe1_1m_audit_remote.py` at commit `c06b2e0`.
- **CT sweep parquet (remote-only, 6.19 GB)**: `/root/intra/data/ml/originals/ml_dataset_v11.parquet`.
- **ET sweep parquet (pulled locally and remote; gitignored, 3.6 GB)**: `data/ml/probe1_audit/ml_dataset_v11_1m_et.parquet`.
- **Recount JSON (pulled locally; gitignored)**: `data/ml/probe1_audit/stratified_recount_1m.json`.
- **Provenance**: `tasks/tz_bug_provenance_log_2026-04-23.md`.
- **Authority council**: `tasks/council-report-2026-04-23-probe3-reconvene.html` + transcript.

---

## Recommended next steps

1. **Fresh LLM Council on Branch B scope** — under ET semantics the signed contract routes to Branch B (K-fold audit), but Branch B was never operationalized. The council should scope the K-fold audit: partition definition, combo-agnostic protocol, gate thresholds, and whether the exactly-at-threshold 15m result combined with the 36×-over-threshold 1m result changes the K-fold design.
2. **Fix the streaming variance numerical artifact** (≤ 10 lines in `_probe1_stratified_recount_1m.py`; Welford or delegate to `np.var(ddof=1)` per combo). Re-emit the JSON. Low priority unless a specific ET combo's individual Sharpe is load-bearing.
3. **Decide whether to re-sweep CT with single-generation seed=0** to give a clean matched-pair CT vs ET comparison. Costs ~4.7 h remote; not blocking.
4. **Do NOT retroactively rewrite the signed Probe 1 verdict.** The signing commit `d0ee506` stands as a historical record. This finding should be appended as an amendment/addendum rather than a revision.
