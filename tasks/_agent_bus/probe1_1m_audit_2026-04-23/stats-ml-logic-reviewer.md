---
from: stats-ml-logic-reviewer
run_id: probe1_1m_audit_2026-04-23
timestamp: 2026-04-23T22:45:00Z
scope_reviewed: Probe 1 1m ET audit — `tasks/probe1_1m_et_audit_finding.md`; streaming recount at `tasks/_probe1_stratified_recount_1m.py`; recount JSON at `data/ml/probe1_audit/stratified_recount_1m.json`; signed verdict/preregistration; v12 `audit_full_gross_sharpe` reference
verdict: unsound
critical_flags: 2
warn_flags: 3
info_flags: 2
cross_references:
  - tasks/probe1_1m_et_audit_finding.md
  - tasks/_probe1_stratified_recount_1m.py
  - tasks/_probe1_gross_ceiling.py
  - tasks/probe1_verdict.md
  - tasks/probe1_preregistration.md
  - scripts/analysis/build_combo_features_ml1_v12.py
  - data/ml/ml1_results_v12/combo_features_v12.parquet
  - data/ml/probe1_audit/stratified_recount_1m.json
  - data/ml/probe1_audit/ml_dataset_v11_1m_et.parquet
---

## Verdict

**UNSOUND.** The headline "N_1.3(1m) = 372 under ET" is based on a **different quantity** than the signed Probe 1 verdict's "N_1.3(1m) ≤ 1 under CT" — they are not comparable, and the recount has therefore not established what it claims. The recount uses the `gross_pnl_dollars` column (dollar PnL at the sweep's risk-sized contract count), while the signed verdict used `audit_full_gross_sharpe` from `build_combo_features_ml1_v12.py:229` which computes Sharpe of `r_multiple × stop_distance × DOLLARS_PER_POINT` (a 1-contract proxy). On the **same ET 1m parquet**, the signed-verdict formula gives **N_1.3 = 0** (max Sharpe 1.027) at either gate=50 or gate=500. Only by swapping in the dollar-PnL-sized column does N_1.3 become 372. Moreover, the dollar-PnL column shows an internal inconsistency — on combo 3595 the friction-implied contract count is 324, but the gross-PnL-implied contract count varies from −2,970 to +3,510 on trades with constant stop — indicating a sizing-pipeline bug or latent column-semantics drift. Until the recount is redone using the same 1-contract gross formula as v12 (and on a comparable CT source), the branch-routing claim is not supported. As a second-order problem: even if we trusted the recount, its stated branch-routing disjunction cites `N_1.3(1m)` as a §3 variable, but the actual signed §3 rule (preregistration §3 line 117-131) disjoins only over {15m, 1h}; 1m is cited in the verdict as "reference" only.

## Methodological Audit

- **Choice of measurement: fails matched-pair principle.** The recount's Sharpe is `mean(gross_pnl_dollars)/std(gross_pnl_dollars) × sqrt(n/years)`. The signed verdict's Sharpe is `mean(rmul*stop*$/pt)/std(...) × sqrt(n/years)` per `build_combo_features_ml1_v12.py:229`. These are the same trades but different pnl series. They produce different Sharpes by a factor of up to ~10× on the same combo (e.g., combo 3595: 9.998 vs 1.027). The two column relationships implicitly depend on contract-count behavior — the recount's measurement carries dollar-sizing variance as part of its signal, whereas the signed measurement is per-contract. Re-auditing a decision by switching the dependent variable is not a re-audit.
- **Streaming-variance formula: correct for the measurement it implements.** I verified on ET combos with n ∈ {17693, 33195, 58957} that `(Σx² − n·μ²)/(n−1)` agrees with `np.std(ddof=1)` to machine precision (diff = 0.00e+00 exact). For gated combos (n ≥ 50) the streaming formula is NOT a source of bias. Catastrophic cancellation shows up only for n ∈ {2, 3, 4} combos with zero realized variance, which are filtered by the MIN_TRADES_GATE.
- **Max-Sharpe reporting bug is cosmetic.** The 4.44e7 max in the ET column is produced by n=3 and n=4 combos where `std_gross_pnl_dollars → 1e-9` due to all trades having near-identical outcomes. These combos are excluded from the N_1.3 count (gate n≥50). The finding doc's caveat 3 is correct that the count is not inflated by these artifacts.
- **Assumptions behind Sharpe-as-ceiling gate**: the preregistration defines "gross Sharpe" implicitly as the pre-friction measure. Neither script enforces that definition; `_probe1_gross_ceiling.py` uses `df['gross_pnl_dollars']` (matching the recount) but was only run on 15m/1h. The 1m signed number came from v12 which used `rmul*stop*$/pt` (1-contract net mislabeled as gross). The whole falsification-evidence chain has inconsistent definitions of "gross Sharpe" across scripts.
- **Alternatives not considered by the finding doc**: (a) compute both measurements side-by-side and document which one the signed §3 gate actually meant; (b) verify that `gross_pnl_dollars` and `r_multiple × stop × $/pt × contracts_from_friction` agree on a random sample of combos before trusting either; (c) rebuild the CT reference by running `build_combo_features_ml1_v12.py` on the ET parquet, which would produce an apples-to-apples `audit_full_gross_sharpe` comparable to the signed verdict; (d) add a +$5/contract conversion to the v12 formula to get "true 1-contract gross" (which gives ET N_1.3 = 129 at gate=50, also above the cited branch-routing gate but roughly 3× smaller than the recount's headline).

## Data Audit

- **Descriptive reality (ET parquet).** 16,384 combos sampled, 13,776 have any trades, 10,949 meet gate n≥50. ET max Sharpe excluding n<50 artifacts: 9.998 (combo 3595, n=33,195) via `gross_pnl_dollars`; 1.027 (via `rmul*stop*$/pt`). Two completely different ceilings.
- **Internal column-consistency defect: CRITICAL.** On combo 3595 (and by inspection 8360, 8375, 11997):
  - Stop distance is constant at 4.625pts.
  - Friction is constant at $1,620 → 324 contracts per trade.
  - Per `param_sweep.py:1030` the expected contract count is `floor(2500/(4.625·2)) = 270`.
  - `r_multiple` formula reconstructs contracts = 270 (via `net/(rm·stop·2)`).
  - Gross-PnL formula reconstructs contracts ranging from **−2,970 to +3,510** (via `gpd/(rm·stop·2+5)`).

  Multiple internal contract-count measures disagree, implying the `gross_pnl_dollars` column was computed with a different (and mathematically inconsistent) contract count than the `friction_dollars`, `net_pnl_dollars`, and `r_multiple` columns. This is not a streaming-variance artifact; it is a column-semantics defect. The recount's headline Sharpe inherits this defect directly.
- **Leakage / contamination risk**: standard v11 sweep output; not a training dataset issue per se.
- **Representativeness (CT-vs-ET)**: the recount compares the CT parquet (now 24,618 combos after multiple appended v11 generations) to the ET parquet (fresh 16,384). Subsetting CT to combo_id < 16,384 yields 13,426 combos — implying **some of CT's early combo_ids have been overwritten or dropped** (expected 16,384 if sweep used an identity combo_id map). Without verifying that combo_id = N in the current CT parquet corresponds to the same parameter tuple as combo_id = N in the ET parquet, per-combo_id comparisons are semantically unanchored. The finding doc's caveat 2 notes this generally but does not verify parameter-tuple equivalence.

## Result Interpretation

- **Statistical significance**: N_1.3 is not a conventional test statistic; it's a count of combos with point-estimated Sharpe ≥ 1.3, so uncertainty is intrinsic to each combo-level Sharpe CI, not the aggregate N_1.3. At n = 220 trades (combo-865 comparable) and YEARS_SPAN = 5.8, σ(Sharpe) ≈ sqrt(YEARS·(1+Sharpe²/2)/n) ≈ 0.24 at Sharpe=1.3 — the gate is far within noise of any single-combo point estimate. N_1.3 is a sample-size-free count; it does not test anything.
- **Practical significance**: N_1.3 = 0 via the signed measurement says "no 1m combo clears an annualized gross Sharpe of 1.3 even before friction, on the ET-corrected parquet." N_1.3 = 372 via the recount says "372 combos have dollar-PnL-sized Sharpes above 1.3" — but that is a different quantity with a column-sizing defect.
- **Multiple-testing / power**: 10,949 combos compared against a fixed 1.3 threshold is a gigantic multiple-testing surface. Under a null of no true edge and Sharpe sampling noise alone, the expected number of combos with empirical Sharpe ≥ 1.3 from a population whose true Sharpe = 0 is nontrivial — for 10k combos with σ ≈ 0.44 (n~500, years 5.8), you'd expect tens to hundreds of combos crossing 1.3 by chance. This is a structural weakness of the preregistration itself (not specific to the ET audit), but it makes the "10 combo" gate a very soft bar. A proper family-level falsification would test at the distribution level (e.g., KS against null Sharpe distribution).
- **What the result licenses you to conclude**:
  - ✗ Cannot conclude "ET N_1.3(1m) = 372" — that is a dollar-PnL-sized Sharpe count on a column with a sizing defect.
  - ✗ Cannot conclude "Branch A cannot fire under ET" — that claim bundles a measurement-mismatched count, a column-defect artifact, and a §3-rule misstatement (which adds 1m to a disjunction that signed §3 never included).
  - ✓ Can conclude "ET 1m produces a meaningfully different population of Sharpes than CT 1m at the column-level, even if the absolute values are disputed."
  - ✓ Can conclude "The signed Probe 1 verdict's `N_1.3(1m) = 1` is a typo for N(≥1.0) = 1" — the v12 parquet has max audit_full_gross_sharpe = 1.108 (single combo, v11_23634) and N(≥1.3) = 0 at gate=500; this is an internally consistent finding.

## Impact on Project Scope

- **If the result stands**: the finding as written would claim the signed bar-timeframe falsification is overturned and Branch B should execute. Given the critical flags below, this is **not** a safe basis for spinning up a combo-agnostic K-fold on 1m. Doing so would sink ~1 day of compute on a premise not yet established.
- **If the result is fragile (which it is)**: the 15m flip CT 9 → ET 10 still stands on its own, per the earlier 15m/1h audit — but that is a 15m result at the exact gate boundary (a known methodological fragility — see `pitfall_exactly_at_threshold.md`). The right move is: (a) fix the sized-gross-vs-1c-gross mismatch, (b) rerun the recount using the signed measurement, (c) decide branch routing on the corrected ET 15m + 1m + 1h triple under the signed §3 rule (which is actually a {15m, 1h} disjunction, not a 3-way).
- **Consistency with prior findings**: the finding doc's claim that "the signed Probe 1 verdict's N_1.3(1m) = 1 is a typo for N(≥1.0) = 1" is correct and I reproduced it directly against the v12 parquet — that caveat stands. But the body of the finding doc moves past this caveat and treats the ET N_1.3 = 372 as compelling evidence, which it is not at this measurement definition.

## Recommended Alternatives (ranked by information value)

1. **Rerun the recount using the v12 measurement formula** (`gross_pnl = rmul × stop_distance_pts × DOLLARS_PER_POINT`, applied to the ET parquet). Cost: trivial (<1 h). Answers: "Under the identical Sharpe measurement protocol used to sign Probe 1, does the ET re-sweep flip Branch A on 1m?" Decision rule: if ET N_1.3(1m) ≥ 10 under this formula, the finding's claim holds; otherwise, the finding's claim is retracted and Branch A stands on 1m. **My preliminary computation says this count is 0 at max Sharpe 1.027.**
2. **Diagnose the `gross_pnl_dollars` column inconsistency on combo 3595 (and peers).** Open `scripts/param_sweep.py:1030-1037`, instrument a test case that prints `(contracts, stop_distance_pts, gross_pnl)` per trade and confirm they satisfy `gross = (exit-entry)·side·contracts·$/pt` with a **single** contract count per trade. If the column is corrupted, the entire recount basis is suspect, but also so are all downstream eval notebooks that consume `gross_pnl_dollars`. This is either a silent sizing-pipeline bug or a latent column-semantics change between sweep generations that was never caught.
3. **Correct the §3 branch-routing quote in the finding doc.** The signed preregistration §3 line 117-131 disjoins over {15m, 1h}; 1m is reference. The finding's routing table (§ "Branch routing under corrected pipeline") invents a three-way disjunction that is not in the signed contract. Under the signed wording, the 15m ET result (N_1.3 = 10 exactly-at-threshold) is the actual routing signal — and that is still fragile per exactly-at-threshold semantics.
4. **Resolve the 1-contract net/gross labeling confusion in v12.** `build_combo_features_ml1_v12.py:229` computes `gross_pnl = rmul * stop * $/pt` and labels this "gross," but `r_multiple` is already net-of-friction per `param_sweep.py:1038`. So v12's `audit_full_gross_sharpe` is actually "Sharpe of 1-contract net." The "true 1-contract gross" requires adding back $5/contract: Sharpe of `rmul*stop*$/pt + 5`. On the ET parquet this gives N_1.3 = 129 at gate=50. Fixing this definition upstream prevents the current confusion from recurring.
5. **Address the family-level multiple-testing weakness in any future preregistration.** 10k+ combos compared against Sharpe ≥ 1.3 has an implicit null-probability mass that depends on YEARS_SPAN and per-combo noise. The "≥ 10 combos" gate is dramatically weaker than it looks. A correctly designed gate would use a distributional test (e.g., KS vs. a permuted-sign null Sharpe distribution) or a p-value-controlled count. Not actionable for the current audit but important if/when the Probe 1 disjunction is revisited.

## Severity Flags

- **CRITICAL**: `gross_pnl_dollars` column on combo 3595 (and likely all v11 swing-stop combos) implies contract counts from −2,970 to +3,510 while friction implies 324 and r-multiple math implies 270. This is an internal column-inconsistency in the sweep parquet, not a recount bug. It invalidates any Sharpe computed from `gross_pnl_dollars` that lacks an explicit sizing verification. Root cause traces to `scripts/param_sweep.py` pipeline — needs dedicated investigation.
- **CRITICAL**: The recount measures a fundamentally different quantity than the signed verdict. Under the signed measurement (`rmul × stop × $/pt`, v12 method), ET N_1.3(1m) = 0, max Sharpe 1.027 at gate=50 (I independently verified by streaming the ET parquet). Branch A would still fire under the signed measurement. The headline "N_1.3(1m) = 372" is therefore not evidence that the signed falsification was wrong.
- **WARN**: The finding doc's branch-routing table ("N_1.3(1m) < 10 AND N_1.3(15m) < 10 AND N_1.3(1h) < 10") contradicts the signed §3 rule at preregistration line 117-131, which is a {15m, 1h} disjunction. 1m was cited as "reference" only in the signed verdict §2 table.
- **WARN**: 15m ET result of exactly 10 is at threshold — known methodological fragility (see `pitfall_exactly_at_threshold.md`). The conclusion "Branch B routes on the corrected pipeline" is a single-coin-flip away from Branch A even on the finding's best reading.
- **WARN**: CT-vs-ET combo_id subsetting is not parameter-tuple-verified. `combo_id = N` in the multi-generation CT parquet may not refer to the same parameter tuple as `combo_id = N` in the fresh ET parquet. Per-ID subsetting only works if seeds and ranges match exactly across generations.
- **INFO**: The signed verdict's "N_1.3(1m) = 1" being a typo for N(≥1.0) = 1 is reproduced directly from the v12 parquet: max `audit_full_gross_sharpe` = 1.108 (v11_23634), N(≥1.3) = 0, N(≥1.0) = 1. Caveat 1 of the finding doc is correct.
- **INFO**: The streaming variance formula `(Σx²−nμ²)/(n−1)` in `_probe1_stratified_recount_1m.py` is arithmetically identical to `np.std(ddof=1)²` for gated combos (diff = 0.00e+00 on independent spot checks). No recount-side numerical bug; the catastrophic-cancellation artifact only manifests for sub-gate combos that are excluded.
