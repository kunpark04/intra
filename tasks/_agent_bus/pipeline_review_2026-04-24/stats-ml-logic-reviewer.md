---
from: stats-ml-logic-reviewer
run_id: pipeline_review_2026-04-24
timestamp: 2026-04-24T08:45:00Z
scope_reviewed: end-to-end probe pipeline audit (Probes 1-4 + Scope D + TZ-audit amendment chain) for residual errors after the 2026-04-23/24 correction cycle
verdict: partially sound
critical_flags: 0
warn_flags: 3
info_flags: 4
cross_references:
  - tasks/probe1_1m_et_audit_finding.md
  - tasks/combo_3595_column_defect.md
  - tasks/probe1_verdict.md
  - tasks/probe1_preregistration.md
  - tasks/probe2_verdict.md
  - tasks/probe3_verdict.md
  - tasks/probe4_verdict.md
  - tasks/scope_d_brief.md
  - tasks/_probe1_gross_ceiling.py
  - tasks/_probe1_recount_v12_formula.py
  - tasks/_probe3_regime_halves.py
  - tasks/_probe3_1h_ritual.py
  - tasks/_probe3_15m_nc.py
  - tasks/_probe3_readout.py
  - tasks/_probe4_readout.py
  - tasks/_scope_d_readout.py
  - scripts/param_sweep.py
  - scripts/analysis/build_combo_features_ml1_v12.py
  - scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py
  - data/ml/probe1_audit/v12_formula_recount.json
  - data/ml/probe1_audit/stratified_recount_et.json
  - data/ml/probe3/regime_halves.json
  - data/ml/probe3/param_nbhd.json
  - data/ml/probe3/15m_nc.json
  - data/ml/probe3/1h_ritual.json
  - data/ml/probe3/readout.json
  - data/ml/probe4/readout.json
  - data/ml/scope_d/readout.json
  - data/ml/probe1_audit/ml_dataset_v11_1m_et.parquet
  - tasks/_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md
---

## Verdict

**PARTIALLY SOUND.** All Group 1 / Group 2 / Group 3 corrections I verified reproduce on independent data checks to machine precision — Amendment 2 of the 1m finding is internally consistent; the combo-3595 column-defect retraction is correct (effective `cost_rt=$6` via `fill_slippage_ticks=1` at `scripts/param_sweep.py:895-901` closes every identity to 0.00e+00); the corrected N_1.3 matrix reproduces on my own stream (1m CT/ET = 0/0 under v12 net, 15m CT/ET = 9/10 under `gross_pnl_dollars`, 1h CT/ET = 4/5 under `gross_pnl_dollars`); the Probe 3 §4.3/§4.4 flips and Probe 4 row-2 inversion are all supported by the post-fix gate JSONs. However I found **one material stale-state issue** (Probe 3 aggregate `readout.json` is pre-TZ-fix and contradicts its per-gate inputs), a **pre-existing sizing-correction error** in Probe 3's W2 post-audit text that the amendment chain has not touched, and several documentation/classifier quirks that are latent but not numerically wrong. Verdict is "partially sound" — the quantitative reasoning in the current (post-Amendment-2) state is correct; the machine records and two documentation items trail the text. No new numerical errors in the gate chain.

## Methodological Audit

### Amendment 2 internal consistency (Group 1 Q1)

Amendment 2 of `tasks/probe1_1m_et_audit_finding.md` correctly distinguishes which formula the signed Probe 1 verdict used for which timeframe:

- **1m row** in the verdict (Sharpe 1.108, N_1.3=1): came from v12's `audit_full_gross_sharpe` which is `rmul × stop × $/pt` — a "1-contract-NET-of-actual-friction" quantity, misnamed "gross" in v12. This is why the "1" was a typo for N(≥1.0)=1; max is 1.108 < 1.3.
- **15m and 1h rows** (Sharpe 1.817 on 9 combos; Sharpe 2.272 on 4 combos): came from `_probe1_gross_ceiling.py` which reads `gross_pnl_dollars` (contract-sized dollar PnL from `param_sweep.py:1034`). See `probe1_verdict.md:22` — "mirrors `scripts/analysis/build_combo_features_ml1_v12.py:254-260`" but the 15m/1h script uses the sweep parquet's `gross_pnl_dollars` column directly, not the v12 formula.

The distinction is correct and load-bearing: Amendment 2 reinstates the 15m CT 9 → ET 10 finding (apples-to-apples under `gross_pnl_dollars`) while keeping the 1m retraction (the 372-count was under `gross_pnl_dollars`; the verdict's 1m row was under the v12 formula; these are different quantities, so the comparison was non-sequitur).

### Combo-3595 retraction independently verified (Group 1 Q2)

Confirmed on the actual ET parquet: combo 3595 has `fill_slippage_ticks=1`, `stop_distance_pts=4.625` constant across 33,195 trades, `friction_dollars=1620` constant → contracts = 1620/6 = **270**. With `cost_rt=$6`:

- `net_pnl_dollars == gross_pnl_dollars − friction_dollars`: max|diff| = 0.00000000
- `r_multiple == net_pnl_dollars / (stop × contracts × 2)`: max|diff| = 0.0000000000
- `gross_pnl_dollars == (r_multiple × stop × 2 + cost_rt) × contracts`: max|diff| = 0.0000000000

The original "defect" (324 vs 270 vs varying) was a back-calc that assumed `cost_rt=$5` uniformly; `scripts/param_sweep.py:895-901` adds `fill_slippage_ticks × 2 × TICK_SIZE × $/pt = $1` for `ticks=1`, so effective `cost_rt=$6`. The retraction text in `tasks/combo_3595_column_defect.md` is accurate.

### Corrected N_1.3 reproduces (Group 1 Q3)

Independently re-ran:

- **1m ET under v12 formula** (`rmul × stop × 2`, stream `ml_dataset_v11_1m_et.parquet`): gated=10,949, N_1.3=0, N_1.0=1, max Sharpe 1.0273. Matches `v12_formula_recount.json` exactly.
- **15m ET under `gross_pnl_dollars`**: 10 combos at Sharpe ≥ 1.3, max 1.9420, gated=1,385. Matches `stratified_recount_et.json` (n_pass_overall: 10 for 15m) exactly.
- **1h ET under `gross_pnl_dollars`**: 5 combos at Sharpe ≥ 1.3, max 2.2723, gated=536. Matches `stratified_recount_et.json` (n_pass_overall: 5 for 1h) exactly.

### The `_probe1_gross_ceiling.py` comment quibble (Group 2 Q1)

Line 11 of `_probe1_gross_ceiling.py`: `"gross_pnl_dollars column already = 1-contract gross PnL per trade"`. This is technically imprecise (the column is **contract-sized**, not 1-contract), but I verified on all 10 passing 15m ET combos that the per-combo contract count is **effectively constant** (stop resolves to a single value per combo via `_resolve_stop_pts` at `param_sweep.py:926-939`; even ATR/swing combos get a median-based fixed resolution). For a fixed-contracts-per-combo series, Sharpe is scale-invariant: mean and std both scale by `contracts`, so `mean/std = gross-per-trade/std-per-trade = 1-contract Sharpe`. For combos 2263/1744/2352/.../408 I computed Sh(contract-sized-gross) and Sh(gross/contracts) and they match to 4 decimal places. The comment is materially correct for this pool even if pedantically loose. **No impact on the signed 15m/1h counts**.

### v12 `audit_full_gross_sharpe` consumers (Group 2 Q2)

Grep confirms the column appears in 14 files:
- `scripts/analysis/build_combo_features_ml1_v12.py:292` — produces it (`full_gross_sharpe`)
- `scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py:75` and `extract_top_combos_v12.py:158` — emit it into JSONs as audit-only; the actual ranking uses `audit_full_net_sharpe` (line 30 of the raw_sharpe script)
- Various eval JSONs and session artifacts — store the value passively
- `tasks/probe1_1m_et_audit_finding.md`, `_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md`, `tasks/_probe1_recount_v12_formula.py` — all correctly document/note the mislabeling

**No consumer ranks, gates, or ships on `audit_full_gross_sharpe`.** The production ranker uses `audit_full_net_sharpe` which has the same mislabel issue (it is actually "net of actual friction minus $5 more", i.e. double-counting friction), but since it's used only for relative ordering — and the $5 constant shifts each combo's mean by a combo-specific amount (contracts × $5) — the ordering is still approximately correct for the eligible-pool combos (MIN_TRADES_GATE=500). This is a pre-existing labeling/double-counting concern separate from the probe pipeline.

### Probe 2/3/4 Sharpe formulas (Group 2 Q3-Q5)

Probe 2 readout (`_probe2_readout.py:52-63`): uses `net_pnl_dollars` column (contract-sized, with actual friction). For combo 865 1h: 220 trades, contract count = floor(2500 / (17.018 × 2)) = **73** (verified: friction/trade = $438 = 73 × $6), net Sharpe 2.895 annualized over `YEARS_SPAN_TEST=1.4799`. Since contracts is constant per combo (fixed stop), Sharpe is scale-invariant → the preregistered gate ($5,000/yr is an absolute dollar gate that DOES depend on sizing; combo-865 clears $124,896/yr easily at 73 contracts). **No formula mismatch.**

Probe 3 regime halves (`_probe3_regime_halves.py:138`): uses `net_pnl_dollars`. Halves (102 + 118 = 220) reproduce the full Sharpe 2.895 between the half-Sharpes (3.235 / 2.580). **Correctly computed; formula matches Probe 2.**

Probe 3 §4.2 param_nbhd + §4.3 15m_nc + §4.4 1h_ritual: all use `net_pnl_dollars` per `_probe3_1h_ritual.py:264` and `_probe3_15m_nc.py` analogue. **Same formula; gate outcomes are consistent with Probe 2.**

Probe 4 readout (`_probe4_readout.py:210`): uses `net_pnl_dollars`. Sharpes 3.551 (combo-1298) and 1.340 (combo-664) + Welch-t 3.851. **Same formula; contracts per combo constant.**

All of these are "contract-sized net Sharpe at the sweep's 5%-of-$50k sizing", which is dimensionally consistent with each other. They are NOT directly comparable to Probe 1's `gross_pnl_dollars` ceiling (also contract-sized but pre-friction), and NOT comparable to v12's `audit_full_gross_sharpe` (1-contract "gross", actually net of actual friction). The three probe-pipeline Sharpes (Probes 2/3/4) are comparable to each other.

### Probe 3 retraction methodological soundness (Group 3 Q1)

`data/ml/probe3/15m_nc.json` at `Apr 23 07:38` shows `n_pass=8, threshold_max=2, gate_pass=false` — this is the post-TZ-fix state. `data/ml/probe3/1h_ritual.json` at `Apr 23 07:38` shows `n_pass=12, threshold_min=8, gate_pass=true`. Under `_probe3_readout.py` §5 routing, current F-count would be **1** → COUNCIL_RECONVENE. **The retraction narrative in `probe3_verdict.md` Amendment 2 matches the regenerated gate JSONs.**

Pre-TZ-fix (under the old `tz_localize("UTC")`), 15m NC was 0/16 PASS (cleanest possible) and 1h ritual was 8/16 exactly-at-threshold PASS. Under corrected CT→ET, 15m NC rescues to 8/16 FAIL (bad — it is a negative control, meaning the 15m signal survives session/exit perturbation, which contradicts the family-level Probe 1 15m falsification framing) and 1h ritual strengthens to 12/16. The flip is structural: the RTH/overnight labels literally inverted, and a signal that was "purely overnight" under the old TZ became "RTH-concentrated" under the corrected one.

**Methodologically sound**: the retraction cascade follows the signed §5.2 branch routing under the regenerated JSON inputs.

### Probe 4 retraction methodological soundness (Group 3 Q2)

`data/ml/probe4/readout.json` (post-fix): combo-1298 SES_1 RTH Sharpe **4.414 on n=77** (+$146,545/yr, `abs_pass: true`); SES_2 GLOBEX Sharpe **0.265 on n=46** (+$6,714/yr, `abs_pass: false` — fails n_trades 46 < 50). §5 row 2 requires `gate_1298_abs_pass AND SES_2_abs_pass AND NOT SES_1_abs_pass AND (SES_2 - SES_1) > 1.0`; under corrected TZ this is `True AND False AND False AND (-4.15 > 1.0 = False)` → row 2 fails three of four conditions. Row 4 fires (both combos pass absolute at SES_0) → COUNCIL_RECONVENE. **Retraction text is correct.**

### Scope D retraction coherence (Group 3 Q3)

`data/ml/scope_d/readout.json` (post-fix, stored at `Apr 23 07:37`) per-combo bucket metrics match the brief's amendment table exactly (865 SES_1 Sharpe 2.635 on 126; 1298 SES_1 Sharpe 4.414 on 77; 664 SES_2a Sharpe 1.174 on 505). The retraction narrative is numerically correct. **One caveat**: the JSON's `regime_label` field for combo 1298 reads `"weak / no clear pattern"` because `_scope_d_readout.py:178-189`'s `_dominance_label` only discriminates between SES_2a-dominance and SES_2b-dominance — it has no label for "SES_1 (RTH) dominance." The brief's amendment manually reinterprets the table as "strongly RTH-concentrated" which is the correct reading of the Sharpes, but a reader looking at `cross_combo_regimes` in the JSON would see `{"865": "weak...", "1298": "weak...", "664": "weak..."}` and conclude nothing meaningful — contradicting the brief's "combo 1298 is strongly RTH-concentrated" interpretation. **Labeling limitation, not a numerical error.**

## Data Audit

- **All column identities verified on combo 3595**: 0.00e+00 error to machine precision at `cost_rt=$6`.
- **1m ET v11 parquet**: 13,776 unique combos, 10,949 gated (n≥50), max Sharpe (v12 formula) 1.0273, N_1.3=0.
- **15m ET v11 parquet**: 2,330 unique combos, 1,385 gated. Under `gross_pnl_dollars`, N_1.3=10 (max 1.942); under v12 formula, N_1.3=2 (max 1.628).
- **1h ET v11 parquet**: 1,094 unique combos, 536 gated. Under `gross_pnl_dollars`, N_1.3=5 (max 2.272); under v12 formula, N_1.3=0 (max 1.115).
- **Probe 2 combo-865 parquet**: 220 test trades, 73 contracts (confirmed via friction $438 = 73 × $6), stop 17.018 constant, $/yr $124,896 at SES_0.
- **Probe 3 gate JSONs**: all regenerated post-fix on Apr 23 07:38; gate_pass booleans consistent with the text narrative.
- **Probe 4 readout.json**: regenerated Apr 23 07:37; `matched_row:4`, `branch: COUNCIL_RECONVENE`.
- **Scope D readout.json**: regenerated Apr 23 07:37; numerical values match the brief's amendment.

Leakage risk: all three probes (2/3/4) have consumed the same 1h test partition. This is already disclosed in the Probe 4 verdict §7.6 and §205 ("the partition as a whole has been 'looked at' three times"); Scope D's brief adds a fourth re-read. No new leakage introduced by the TZ-fix cascade.

## Result Interpretation

What the data now says under the current (post-Amendment-2 + post-a38d3c5) state:

- **Probe 1 Branch A (family sunset) is TZ-sensitive at 15m**: CT 9 vs ET 10 under `gross_pnl_dollars` (exactly-at-threshold; known fragility per `memory/pitfall_exactly_at_threshold.md`). Under the v12 1-contract-net formula, both CT and ET give 2 — no crossing. The two formulas answer different questions: `gross_pnl_dollars` asks "is the pre-friction dollar edge detectable at contract-size sizing," while v12 asks "is the 1-contract net-of-friction Sharpe high." The signed verdict used `gross_pnl_dollars` for 15m/1h, so at-threshold fragility applies.
- **Probe 2 (combo-865 1h PASS) stands** — session-agnostic result, Sharpe 2.895 on 220 test trades.
- **Probe 3 (combo-865 4-gate)**: F=1 under the current gate JSONs, branch should be COUNCIL_RECONVENE. The text retraction in `probe3_verdict.md` Amendment 2 says this, but the aggregate `data/ml/probe3/readout.json` is **stale** at pre-fix F=0, branch=PAPER_TRADE. See WARN flag.
- **Probe 4**: branch is COUNCIL_RECONVENE under corrected TZ (§5 row 2 condition inverts; row 4 fires). Absolute gates (1298 Sharpe 3.551, 664 Sharpe 1.340, Welch-t 3.851) are session-agnostic and stand. B1 session-structure sweep authorization is withdrawn.
- **Scope D**: combos diverge — 865 RTH-leaning (2.64 vs 1.47), 1298 strongly RTH-concentrated (4.41 vs 0.53), 664 weakly overnight-leaning (1.17 vs 0.54). No shared overnight structure.

Multiple-testing exposure: partition-reuse count is now 4 (Probes 2, 3, 4, Scope D) on the 1h test partition; combo-1298's RTH Sharpe 4.41 on n=77 is the headline single result but was selected from a 1500-combo Probe 1 pool and re-measured 4×, so selection + reuse discounts compound. No action-level ship gate depends on the 4.41 Sharpe alone, so this is well-calibrated.

## Impact on Project Scope

- **If the current (post-Amendment-2) readings stand**: No PAPER_TRADE authorization exists anywhere in the project. Probe 3's PAPER_TRADE is withdrawn (COUNCIL_RECONVENE is the correct branch); Probe 4's SESSION_CONFOUND → B1 session-structure-sweep is withdrawn. The project is correctly in a "no authorized deployment path" state with a genuine RTH-concentrated single-combo observation on combo-1298 (Sharpe 4.41 on n=77, one 1500-combo selection × 4-reuse discount applied).
- **If the stale Probe 3 `readout.json` is used by an automated downstream**: it would report F=0, branch=PAPER_TRADE, posterior=0.91. The text narrative has moved past this but the machine record has not. A downstream agent or human reading the JSON would get the obsolete answer.
- **Consistency with prior findings**: the correction cycle has internally reconciled Probes 1 (TZ-sensitive, 15m exactly-at-threshold) / 2 (unchanged) / 3 (retracted to COUNCIL_RECONVENE) / 4 (retracted to COUNCIL_RECONVENE) / Scope D (combos diverge). The narrative is coherent; the machine record on Probe 3 alone lags.

## Recommended Alternatives (ranked by information value)

1. **Regenerate `data/ml/probe3/readout.json`** by running `python tasks/_probe3_readout.py`. Cost: trivial (<1 s). This will read the current per-gate JSONs (which are post-fix) and emit the correct aggregate (F=1, branch=COUNCIL_RECONVENE, posterior≈0.05). Decision rule: if any downstream reads Probe 3's machine record as ground-truth, the current stale state is a footgun; the text retraction is clear but the JSON is not.

2. **Fix the Probe 3 W2 correction text** in `tasks/probe3_verdict.md:248-255` (and line 430). The "~15 contracts at $500 risk" is wrong for the sweep engine, which uses `fixed_equity × RISK_PCT = $50,000 × 0.05 = $2,500` per trade → **~73 contracts** for combo-865 (stop=17.018, $/pt=2). I verified: `friction_dollars=438 = 73 × $6/contract` on the actual Probe 2 parquet. The "fixed_dollars_500" convention in CLAUDE.md is an EVAL-NOTEBOOK sizing, not a sweep-engine sizing. No gate outcome is affected (Sharpe is scale-invariant per constant-contracts-per-combo), but Phase E1 council deliberations about "MNQ contract capacity" would be grossly mis-scoped at 15 vs 73. This is a documentation error, not a data error.

3. **Fix the `regime_label` classifier** in `tasks/_scope_d_readout.py:178-189` to recognize SES_1-dominance. Current logic only labels "SES_2a dominates / SES_2b dominates / mixed / weak". Combo 1298 with SES_1 Sharpe 4.41 / SES_2a 0.53 / SES_2b -0.78 gets "weak / no clear pattern" — contradicting the brief's amendment. Minimal change: add a branch like `if sharpe_1 > 1.3 AND sharpe_1 > 2 × max(sharpe_2a, 0): return "SES_1 (RTH) dominates"`. Alternatively, regenerate `scope_d/readout.json` and accept that the regime_label field is merely a SES_2-split characterization, not a full session-regime classifier — and update the brief's amendment to say that explicitly. Low priority.

4. **Fix the misleading comment in `_probe3_regime_halves.py:120`** (`"Bar times are naive UTC from the CSV; compare naive"`). The bars are naive **CT**, not UTC. The split operates naively so no numerical error results, but the comment contradicts what the TZ-fix cascade established. One-line change.

5. **Fix the loose comment in `_probe1_gross_ceiling.py:11`** (`"gross_pnl_dollars column already = 1-contract gross PnL per trade"`). For the 15m/1h pool this is materially correct because stops resolve to a single value per combo, but a reader who didn't verify would assume it's universally true. One-line precision fix: `"gross_pnl_dollars is contract-sized; for fixed-contracts-per-combo series (true in this pool, each combo has a single resolved stop), Sharpe is scale-invariant and equals the 1-contract Sharpe."`

6. **Consider a fresh preregistration for any future Probe 1 re-audit** that explicitly names the Sharpe formula. The current archaeology of "which formula did the signed verdict actually use for which TF" is a source of confusion that a future preregistration should preempt. Not urgent.

## Severity Flags

- **WARN**: `data/ml/probe3/readout.json` timestamp `Apr 22 01:11` predates the TZ-fix commit `a38d3c5` (Apr 23 07:54) and shows `F:0, branch:PAPER_TRADE, posterior:0.91`. The per-gate inputs (`15m_nc.json`, `1h_ritual.json`) were regenerated on Apr 23 07:38 and have `gate_pass:false` / `gate_pass:true` respectively. Re-running `python tasks/_probe3_readout.py` would emit `F:1, branch:COUNCIL_RECONVENE, posterior≈0.05`. Text narrative in `probe3_verdict.md` Amendment 2 already documents this transition, but the machine record is inconsistent. Any downstream automation that reads `readout.json` as source-of-truth will get the obsolete answer.

- **WARN**: `tasks/probe3_verdict.md` W2 correction at lines 248-255 (and recap at line 430) states combo-865 runs on "~15 MNQ contracts at fixed $500 risk". The sweep engine uses $2,500 risk per trade (5% of $50k equity per `scripts/param_sweep.py:94-95`, `scripts/param_sweep.py:1030`), giving **73 contracts** for combo-865 (verified: friction per trade $438 = 73 × $6 with `fill_slippage_ticks=1`). The W2 text conflates sweep-engine sizing with the eval-notebook `fixed_dollars_500` convention. No gate outcome affected; but Phase E1 council deliberations that inherit W2's "~15 contracts" framing would be mis-scoped by ~5×.

- **WARN**: `tasks/_scope_d_readout.py:178-189`'s `_dominance_label` classifier has no "SES_1 (RTH) dominates" case. Combo 1298 (Sharpe 4.41 RTH / 0.53 overnight / -0.78 halt) is classified "weak / no clear pattern" in the JSON, contradicting the brief's amendment text "strongly RTH-concentrated". The numbers in the JSON are correct; the human-readable summary field is not.

- **INFO**: Comment at `tasks/_probe3_regime_halves.py:120` still says `"Bar times are naive UTC from the CSV"` — factually wrong (bars are CT per `scripts/data_pipeline/update_bars_yfinance.py:37`). The split logic is naive-on-both-sides so no numerical error; documentation-only.

- **INFO**: Comment at `tasks/_probe1_gross_ceiling.py:11` says `"gross_pnl_dollars column already = 1-contract gross PnL per trade"`. Contract-sized is more precise; Sharpe happens to be scale-invariant for this pool because stops resolve to a single value per combo. No error in the N_1.3 counts produced.

- **INFO**: `YEARS_SPAN_TEST=1.4799` inherited from Probe 3 is slightly over-stated vs the actual 1h test window (2024-10-30 21:00 → 2026-04-08 19:00 = 1.4371 years). This is already disclosed in `probe4_verdict.md` §"YEARS_SPAN_TEST inherited conservative bias" and corresponds to a ~1.1-1.5% under-estimate of annualized Sharpe. Pre-existing, not introduced by the TZ cascade.

- **INFO**: v12's `audit_full_gross_sharpe` is actually "1-contract net of actual friction" (because `r_multiple` at `param_sweep.py:1038` is already net). v12's `audit_full_net_sharpe` double-counts friction (subtracts $5 on top of rmul-derived net). No production ranker uses `audit_full_gross_sharpe` for ranking; `extract_top_combos_by_raw_sharpe_v12.py` ranks on `audit_full_net_sharpe`. The double-counted friction shifts the per-trade mean by `contracts × $5` which is combo-specific, so ranking is approximately (but not exactly) correct on the MIN_TRADES_GATE=500 eligible pool. This is a pre-existing labeling issue orthogonal to the probe pipeline.

