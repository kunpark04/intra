# Probe 4 Verdict — Cross-TF-Coherence Property Test Resolves to SESSION_CONFOUND

> 🛑 **RETRACTED 2026-04-23 UTC — verdict rerouted SESSION_CONFOUND → COUNCIL_RECONVENE.**
> The same TZ bug that retracted Probe 3 also inverted the §4.4 session
> decomposition and §5 row 2 condition here. Under corrected TZ: combo-1298 is
> strongly **RTH-concentrated** (Sharpe **4.414** on n=77 RTH trades,
> +$146,545/yr) and SES_2 overnight fails the absolute gate (Sharpe 0.265 on
> n=46). Combo-664 is weakly overnight-leaning (Sharpe 1.251 on n=537, just
> under the 1.3 gate). The §5 row 2 condition `SES_2 abs PASS AND SES_1 abs
> FAIL AND Δ>1.0` flips to FALSE — both components invert. Row 2 does not fire;
> row 4 (both-pass adjudication) fires instead → branch = **COUNCIL_RECONVENE**.
> The three absolute-gate Sharpes (1298 3.55, 664 1.34) and Welch-t (3.851)
> are session-agnostic and stand unchanged. The downstream binding "B1
> session-structure sweep authorized" is **retracted** — there is no
> family-level overnight pattern to sweep. See Amendment 2 at the bottom
> of this document.

**Date**: 2026-04-23 UTC
**Probe**: Cross-TF-coherence property test (Path B, B2 second-combo carve-out) — combos 1298 and 664 on 1h test partition (2024-10-22 → 2026-04-08)
**Preregistration**: `tasks/probe4_preregistration.md`
  - Original signing: `432fb3d`
  - Amendment 1 (§4.4 SES_2 complement-of-RTH): `11ae5f4`
  - Scripts: `173372d`
**Readout script**: `tasks/_probe4_readout.py`
**Machine record**: `data/ml/probe4/readout.json`
**Remote launcher**: `tasks/_run_probe4_remote.py`

---

## Bottom line

**Branch = SESSION_CONFOUND per preregistration §5 row 2.** 1298 passes absolute, SES_2 passes absolute, SES_1 fails absolute, and `(SES_2 net Sharpe − SES_1 net Sharpe) = 4.22 > 1.0`. Row 2 fires before rows 3 / 4 under ordered-match per §5.1's prospective ordering disclosure — this exact scenario was named in the prereg before the data landed.

Downstream binding per §5.1 (signed at `432fb3d`, unchanged by Amendment 1): **B1 preregistration is the next authoring step — a session-structure sweep across multiple combos, not a single-combo carve-out. B2 parks.**

Mechanically-applied gates:

| Gate | Threshold | Observed | Margin | Status |
|---|---|---:|---:|:---:|
| §4.1 combo-1298 abs (SES_0) | net Sharpe ≥ 1.3, n ≥ 50, $/yr ≥ $5k | 3.551 · 123 · $153,259 | +2.25 on Sharpe | **PASS** |
| §4.2 combo-664 abs (SES_0) | net Sharpe ≥ 1.3, n ≥ 50, $/yr ≥ $5k | 1.340 · 780 · $83,426 | +0.04 on Sharpe | **PASS** |
| §4.3 Welch-t (PRIMARY) | t ≥ 2.0 one-tailed, n ≥ 50 per arm | t = **3.851** | +1.85 over threshold | **PASS** |
| §4.4 SES decomposition (readout) | non-gate-bound | see §4.4 section | — | — |

All three gates pass. Under rows 1 and 3 in isolation this would have routed PROPERTY_VALIDATED; under rows 3 and 4 in isolation this would have routed COUNCIL_RECONVENE (both-pass adjudication). But row 2 (SESSION_CONFOUND) evaluates first and its conditions hold, so the binding verdict is SESSION_CONFOUND.

All PASS flags sourced directly from `data/ml/probe4/readout.json`; verdict applied by `tasks/_probe4_readout.py::route_branch` against §5 table.

---

## §4.1 — Absolute gate on combo-1298 (SES_0 baseline)

| Metric | Value | Threshold |
|---|---:|---:|
| `n_trades` | **123** | ≥ 50 |
| `net_sharpe` | **3.551** | ≥ 1.3 |
| `net_dollars_per_year` | **+$153,259** | ≥ $5,000 |
| `mean_pnl` / trade | +$1,843.97 | — |
| `std_pnl` / trade | $4,734.17 | — |

Combo-1298 clears all three sub-gates with substantial margin. The mean per-trade net PnL of $1,844 is ~3.3× Probe 2's combo-865 per-trade mean (~$568/trade at 1h), despite 1298 being a different parameter realization with only microstructure overlap (Δ entry_timing/slippage/cooldown = -1/+1/0). This strong absolute result is what would have routed PROPERTY_VALIDATED in isolation (row 3) but is preempted by row 2 under ordered-match.

## §4.2 — Absolute gate on combo-664 (SES_0 baseline)

| Metric | Value | Threshold |
|---|---:|---:|
| `n_trades` | **780** | ≥ 50 |
| `net_sharpe` | **1.340** | ≥ 1.3 |
| `net_dollars_per_year` | **+$83,426** | ≥ $5,000 |
| `mean_pnl` / trade | +$158.28 | — |
| `std_pnl` / trade | $2,712.43 | — |

Combo-664 ALSO clears absolute, despite failing Probe 1's training-partition pre-gate at gross Sharpe 1.200 (below 1.3). Margin is slim: net Sharpe 1.340 is +0.04 over the 1.3 gate. This is a genuine held-out result — the test partition was untouched when 664's identity was selected into Probe 4. 664 carrying a standalone edge on the held-out partition is the unexpected finding and is discussed in the interpretation section below.

**Absolute-gate both-pass interaction**: §5 row 4 (both combos pass absolute) would have routed COUNCIL_RECONVENE for adjudication. It did not fire because row 2 evaluated first. The prereg §5.1 ordering disclosure preemptively explains this:

> *"if combo-1298 and combo-664 both PASS absolute AND combo-1298's edge concentrates in SES_2 only (SES_2 abs PASS, SES_1 abs FAIL, (SES_2 Sharpe − SES_1 Sharpe) > 1.0), the case routes to SESSION_CONFOUND (row 2) rather than to COUNCIL_RECONVENE both-pass adjudication (row 4). This is intentional — a session-confounded signal is diagnostically more informative than a both-pass adjudication, and the binding session-structure interpretation should not be suppressed by concurrent 664 absolute pass."*

Prereg integrity honored: the ordering outcome was signed before the data landed.

## §4.3 — Welch-t primary gate

Per-trade net PnL distributions compared between 1298 (n=123) and 664 (n=780), formula per prereg §4.3:

`t = (mean_1298 − mean_664) / sqrt((var_1298 / n_1298) + (var_664 / n_664))`

| Term | Value |
|---|---:|
| `mean_1298` | $1,843.97 |
| `mean_664` | $158.28 |
| `mean_diff` (numerator) | **+$1,685.68** |
| `var_1298` | 22,412,342 |
| `var_664` | 7,357,275 |
| denominator | $437.78 |
| **Welch-t** | **3.851** |
| Threshold | ≥ 2.0 |
| Insufficient-n defense | no (both arms ≥ 50) |

Welch-t of 3.851 clears the 2.0 threshold comfortably (~92% above threshold). The primary gate passes — 1298's per-trade edge is statistically much stronger than 664's despite 664 also being in-the-money on aggregate Sharpe. 1298's mean per-trade net PnL is **11.6×** 664's.

**Economic reading**: 1298's edge is concentrated and high per-trade (~$1,844/trade, 123 trades/1.48y); 664's edge is diffuse and thin per-trade (~$158/trade, 780 trades/1.48y). Gross Sharpe parity does NOT imply per-trade parity.

## §4.4 — Session decomposition (mandated readout)

Per Amendment 1 (`11ae5f4`), sessions defined as:
- **SES_0**: all sessions (full 24h)
- **SES_1**: ET minute ∈ [570, 960) = 09:30-16:00 ET = RTH
- **SES_2**: complement of RTH = [0, 570) ∪ [960, 1440) = overnight + post-RTH + halt

Implementation: post-hoc ET-minute partitioning via `tz_convert("America/New_York")` on `entry_bar_idx` against `data/NQ_1h.parquet` test partition (engine `session_filter_mode=0` for both runs; the C1 fix from pass-4 review).

### Combo-1298 session decomposition

| Session | n_trades | net_sharpe | net_$/yr | mean_pnl | abs_pass |
|---|---:|---:|---:|---:|:---:|
| **SES_0** all | **123** | **3.551** | **+$153,259** | +$1,844 | ✅ |
| SES_1 RTH | 23 | -0.173 | -$3,020 | -$194 | ❌ |
| **SES_2** ex-RTH | **100** | **4.049** | **+$156,279** | +$2,313 | ✅ |

**81% of 1298's trades land in SES_2**, and SES_2 carries the entire edge (Sharpe 4.05). SES_1 is a net loss. The SES_2 → SES_1 Sharpe gap is **4.22** — well above the 1.0 threshold in §5 row 2 condition.

### Combo-664 session decomposition

| Session | n_trades | net_sharpe | net_$/yr | mean_pnl | abs_pass |
|---|---:|---:|---:|---:|:---:|
| **SES_0** all | **780** | **1.340** | **+$83,426** | +$158 | ✅ |
| SES_1 RTH | 169 | 0.144 | +$4,169 | +$36.50 | ❌ |
| **SES_2** ex-RTH | **611** | **1.437** | **+$79,257** | +$192 | ✅ |

**78% of 664's trades land in SES_2**, and SES_2 carries 95% of the dollar edge (+$79,257 / +$83,426). SES_1 is marginally positive but far below the 1.3 Sharpe gate and the $5k/yr dollar gate.

### Cross-combo session pattern (the key finding)

Both 1298 and 664 share the same overnight concentration:
- ~80% of trades in SES_2 (ex-RTH window including overnight, post-close, and settlement halt)
- SES_2 Sharpe comfortably clears 1.3 for both combos (4.05 for 1298, 1.44 for 664)
- SES_1 (RTH) fails the absolute gate for both combos

The signal under test — "Z-score mean-reversion on 1h NQ" — is overwhelmingly an **overnight session phenomenon** for these two parameter realizations, not a full-day phenomenon. This is structurally consistent with Probe 3's combo-865 finding (overnight/RTH per-trade ratio ~3.45× on 1h; §4.4 of Probe 3) though Amendment 1's SES_2 redefinition means the Probe 4 numbers are not directly comparable to Probe 3's narrow-wraparound SES_2 (cross-probe reasoning requires sub-window partitioning).

---

## §5 — Branch routing

| # | Condition | Evaluation | Fires? |
|---|---|---|:---:|
| 1 | `gate_1298_abs_pass` == FALSE AND Welch-t < 2.0 | False AND False | No |
| 2 | `gate_1298_abs_pass` == TRUE AND SES_2 abs PASS AND SES_1 abs FAIL AND (SES_2 Sharpe − SES_1 Sharpe) > 1.0 | True AND True AND True AND (4.22 > 1.0) | **YES → SESSION_CONFOUND** |
| 3 | — (row 2 already fired) | not evaluated | — |
| 4 | — (row 2 already fired) | not evaluated | — |
| 5 | — (row 2 already fired) | not evaluated | — |

Row 2 is satisfied. Verdict: **SESSION_CONFOUND**. Narrow-miss flag: False (1298 abs PASS, not a [1.1, 1.3) miss).

### §5.1 downstream bindings

- **B1 preregistration becomes the next authoring step**, specifically a session-structure sweep across multiple combos (not a single-combo carve-out). The 1h test-partition finding generalizes beyond combo-865: both 1298 and 664 show the same SES_2-concentrated pattern, suggesting the signal is a session-structural phenomenon that a broader sweep should characterize.
- **B2 parks.** Path B no longer pursues second-combo carve-outs at this stage; the session-structure probe is logically prior.
- **No automatic memory update for Probe 3.** Probe 3's combo-865 verdict (PAPER_TRADE at F=0, commit `b68fe62`) stands unchanged. Combo-865's PAPER_TRADE authorization is not retroactively modified.
- **Combo-1298 is NOT deployed** under this branch. Per §6.6: PROPERTY_VALIDATED authorizes B1 as a follow-on; it does not authorize 1298 trading. SESSION_CONFOUND authorizes B1 even more pointedly, with the same deployment prohibition.
- **Probe 1 §7.6 family-level falsification remains terminal.** Even with 1298 and 664 both showing 1h SES_2 edges, the evidence does NOT re-open the bar-timeframe axis. Per §7.5 of the prereg: 1298 and 865 sit in the same parameter basin (Δ=1 microstructure); the pair plus 664 (microstructure-identical to 865) is not N_1.3 ≥ 10 and does not satisfy Probe 1's family-level gate.

---

## Interpretation

### What Probe 4 found

1. **1298's edge is real and strong on the held-out 1h partition** (Sharpe 3.55, +$153k/yr, 123 trades). Welch-t 3.85 confirms it statistically dominates 664.
2. **664's edge is also real on the held-out 1h partition**, despite failing Probe 1's training-partition pre-gate. This was the unexpected finding.
3. **Both edges are overwhelmingly SES_2-concentrated.** 80% of 1298's trades, 78% of 664's — both combos' SES_1 (RTH) results fail absolute. The signal is overnight-structural, not full-day.
4. **Cross-TF-coherence property status**: the admission of combo-865 to Probe 2 via "top-10 on both 15m and 1h" produced a real 1h signal both for 865 and for 1298 (the rank-1 on 1h), AND for 664 (rank-6 on 1h, did not pass the property pre-gate but still passes held-out absolute). The property isn't pure noise — but the probe cannot distinguish whether the property "works" because of cross-TF coherence or because of shared-overnight-session-structure across the parameter basin. That's what B1 is for.

### What Probe 4 does NOT imply

- Does not unfalsify Probe 1 family-level sunset (still terminal at N_1.3 = 4 on 1h; Probe 4's two-combo evidence is structurally one observation per §7.5 for 1298+865, plus one independent observation for 664 — does not clear the N_1.3 ≥ 10 family gate).
- Does not change Probe 3's combo-865 PAPER_TRADE authorization (commit `b68fe62`; not executed per user's Path B selection).
- Does not authorize combo-1298 paper trading, live trading, or any form of capital-adjacent deployment.
- Does not refute the cross-TF-coherence property — it surfaces that the property co-varies with session structure in a way the original B2 test could not cleanly separate.

### Next probe: B1 (session-structure sweep)

The binding next step is B1 preregistration. Scope should be a multi-combo sweep (~20-50 combos from the Probe 1 pool, pre-gated by some volume threshold) with mandatory SES_0/SES_1/SES_2 decomposition as primary gates (not non-gate-bound readouts). If the SES_2-concentration pattern replicates across a broader combo set, the family-level interpretation shifts from "Z-score mean-reversion on 1h" to "overnight session-structure inefficiency on 1h, detectable via Z-score entries." That's a materially different thesis with different deployment implications (e.g., broker must preserve overnight GLOBEX access; sizing policy must respect the concentrated-session risk profile).

B1 requires a fresh LLM Council pre-preregistration to scope multiplicity, gate thresholds, and session decomposition binding. Per `feedback_council_methodology.md`: Rule 1 (stage naming) and Rule 2 (P(PASS|H_k) per gate) framing are mandatory.

---

## Methodology disclosures

### Amendment 1 (SES_2 complement-of-RTH)

Per the amendment at commit `11ae5f4`, SES_2 is defined as complement-of-RTH (`et_min ∈ [0, 570) ∪ [960, 1440)`) rather than Probe 3's narrow wraparound (`[1080, 1440) ∪ [0, 570)`). The 16:00-18:00 ET window (post-RTH continuation + CME settlement halt) is in Probe 4's SES_2 but NOT in Probe 3's SES_2.

**Practical impact of this choice on the verdict**: unknown without sub-window partitioning. Probe 4's SES_2 Sharpe of 4.05 for combo-1298 reflects a 100-trade aggregate; if a non-trivial fraction of those trades land in 16:00-18:00 ET, the "pure overnight" per-trade ratio could be lower than reported. The amendment's cross-probe comparability caveat stands: **Probe 4 SES_2 numbers are NOT directly comparable to Probe 3's reported 3.45× overnight/RTH ratio on combo-865**. Any B1 preregistration that tries to anchor on cross-probe SES_2 reasoning must partition further by sub-window.

### YEARS_SPAN_TEST = 1.4799 (inherited conservative bias)

Per inherited agent memory `feedback_years_span_cross_tf.md`: the `YEARS_SPAN_TEST` constant inherited from Probe 3 implies 540.5 days; the actual test partition 2024-10-22 → 2026-04-08 is 533 calendar days = **1.4593 years** (7.5-day discrepancy). All annualized metrics (net_dollars_per_year especially) inherit a ~1.3% conservative bias — reported numbers are slightly LOWER than the true annualized values would be.

**Disclosure**: not "fixed" silently per inherited memory policy. The true-window-span recalibration would add ~$2,000/yr to combo-1298's +$153,259 and ~$1,100/yr to combo-664's +$83,426 — neither materially moves the absolute-gate outcome, and the Welch-t gate is scale-invariant to this constant.

### Partition-reuse caveat (§7.6 of prereg)

The 1h test partition has now been consumed by Probe 2 (combo-865 PASS), Probe 3 (combo-865 4-gate PASS), and Probe 4 (combos 1298 + 664). Per prereg §7.6: combos 1298 and 664 have never been run against this partition before Probe 4, so no strict data leakage — but the partition as a whole has been "looked at" three times. The primary Welch-t gate (§4.3) is differential between 1298 and 664, which cancels common-factor window favorability; the absolute gates do not. The SESSION_CONFOUND verdict leans more on the differential Welch-t and the session-structure pattern (both combos → SES_2-concentrated) than on the absolute Sharpe levels, which is the appropriate epistemic posture given the partition-reuse caveat.

### Scope integrity

No deviations from the signed prereg (beyond Amendment 1 which was user-authorized via sign-off at `11ae5f4`). Gate thresholds, branch routing, irrevocable commitments, and execution plan all applied mechanically per the frozen contract.

---

## Cross-references

### Prereg (binding contract)
- `tasks/probe4_preregistration.md` — original signing `432fb3d`, amendment 1 `11ae5f4`
- `tasks/_agent_bus/probe4_2026-04-22/` — 9 review/handoff artifacts across 4 passes

### Execution
- `tasks/_probe4_run_combo.py` — per-combo backtest runner (173372d)
- `tasks/_probe4_readout.py` — §4.3 Welch-t + §5 branch routing (173372d)
- `tasks/_run_probe4_remote.py` — paramiko orchestrator (173372d)

### Machine record
- `data/ml/probe4/readout.json` — canonical verdict ground truth
- `data/ml/probe4/combo{1298,664}_SES_0.json` — per-run aggregates
- `data/ml/probe4/combo{1298,664}_SES_0_trades.parquet` — per-trade PnL + entry_bar_idx
- `data/ml/probe4/run_status.json` — wrapper exit-code status

### Evidence chain predecessors
- `tasks/probe1_verdict.md` — family-level sunset (Branch A)
- `tasks/probe2_verdict.md` — combo-865 single-combo PASS on 1h
- `tasks/probe3_verdict.md` — combo-865 4-gate robustness PASS (F=0, PAPER_TRADE, not executed)

### Council authority (for B2 scope)
- `tasks/council-report-2026-04-22-b2-scope.html`
- `tasks/council-transcript-2026-04-22-b2-scope.md`

### Relevant memory
- `memory/project_probe4_signed.md` — signed state + amendment + scripts lineage
- `memory/project_probe3_combo865_pass.md` — predecessor verdict
- `memory/feedback_council_methodology.md` — Rule 1 + Rule 2 framing (applied to B1 council)
- `memory/feedback_remote_execution_only.md` — compute locality
- `memory/feedback_years_span_cross_tf.md` — conservative-bias disclosure policy

---

## Status

**UNSIGNED DRAFT** as of 2026-04-23 UTC.

Awaits: (1) user review of the verdict text and numbers, (2) user explicit authorization to commit. Once authorized, this document goes into a verdict commit mirroring Probe 3's `18a22ee` pattern. No downstream action (B1 preregistration, memory update on pivot) fires until the verdict is signed.

*Note: this "UNSIGNED DRAFT" status is stale — the verdict was signed at commit `c419391` and addendum `ec60bf0` on 2026-04-23 UTC, prior to the Amendment 2 retraction below.*

---

## Amendment 2 — Timezone bug retraction (2026-04-23 UTC)

### Summary

`tasks/_probe4_readout.py:129` contained `ts.dt.tz_localize("UTC")` on
`data/NQ_1h.parquet` timestamps, which are actually naive **Central Time**
(Barchart vendor export — see `scripts/data_pipeline/update_bars_yfinance.py:37`).
Localizing CT as UTC then converting to ET shifts every bar by ~5–6 hours,
which inverts the SES_1 (RTH) and SES_2 (GLOBEX) wall-clock labels.

Discovered 2026-04-23 UTC by stats-ml-logic-reviewer during paper-trade-readiness
review. Root cause + propagation trace:
`lessons.md` `2026-04-23 tz_bug_in_session_decomposition`.

### Unaffected (session-agnostic — stand as signed)

- **§4.1 combo-1298 absolute gate (SES_0)**: Sharpe **3.551**, n **123**, **+$153,259/yr** — **PASS**.
- **§4.2 combo-664 absolute gate (SES_0)**: Sharpe **1.340**, n **780**, **+$83,426/yr** — **PASS**.
- **§4.3 Welch-t primary gate**: t = **3.851** — **PASS**. Pooled per-trade statistic, session-agnostic.

### Inverted (session-dependent — differ materially)

Corrected §4.4 readout under CT → ET (run 2026-04-23 UTC after the fix):

| Combo | Bucket | Pre-fix Sharpe | Pre-fix n | Post-fix Sharpe | Post-fix n | Post-fix $/yr |
|---|---|---:|---:|---:|---:|---:|
| 1298 | SES_1 RTH | −0.173 | 23 | **+4.414** | **77** | **+$146,545** |
| 1298 | SES_2 GLOBEX | +4.049 | 100 | +0.265 | 46 | +$6,714 |
| 664 | SES_1 RTH | +0.144 | 169 | +0.540 | 243 | +$18,822 |
| 664 | SES_2 GLOBEX | +1.437 | 611 | +1.251 | 537 | +$64,604 |

**Combo-1298 is RTH-concentrated**: 77 of 123 trades (63%) land in RTH with a
Sharpe of 4.414 — the cleanest single readout in the project. SES_2 overnight
Sharpe fails absolute (0.265 < 1.3).

**Combo-664 is weakly overnight-leaning**: 537 of 780 trades (69%) land
overnight with Sharpe 1.251 — just under the 1.3 gate. Neither bucket clears
absolute alone.

### §5 branch routing under corrected gates

Row 2 condition: `gate_1298_abs_pass AND SES_2_abs_pass AND NOT SES_1_abs_pass
AND (SES_2 Sharpe − SES_1 Sharpe) > 1.0`. Under corrected TZ this becomes
`True AND False AND False AND ...` — **row 2 does NOT fire**. Row 3 requires
664 abs FAIL — that also doesn't hold (664 still passes SES_0 absolute). Row 4
fires: both combos pass absolute.

**New branch: COUNCIL_RECONVENE (row 4, both-pass adjudication).**

The verdict was signed as SESSION_CONFOUND at commit `c419391`; under corrected
TZ, the signed §5 branch-routing table would have routed to COUNCIL_RECONVENE.

### What is retracted

- **SESSION_CONFOUND branch.** The session-purity condition that row 2
  required (SES_2 pass + SES_1 fail + delta > 1.0) is inverted under corrected
  TZ: SES_1 (RTH) passes for combo-1298 and SES_2 fails. Row 2 cannot fire.
- **"Both combos ~80% SES_2-concentrated" structural finding.** Under
  corrected TZ, combo-1298's trade distribution is 63% RTH / 37% overnight
  (not 19% / 81%); combo-664's is 31% RTH / 69% overnight. The two combos
  have **opposite** session preferences, not shared overnight concentration.
- **B1 preregistration authorization ("session-structure sweep across
  multiple combos is the next authoring step").** The B1 downstream binding
  was predicated on the cross-combo shared-overnight pattern. Under corrected
  TZ, there is no shared overnight pattern to investigate at the family
  level. B1 as scoped by `tasks/council-report-2026-04-23-b1-scope.html`
  is obsolete; the council's "don't launch B1" verdict and its proposed
  paper-trade alternative are both obsolete too (paper-trade preregistration
  is no longer authorized because Probe 3 has been retracted; see Probe 3
  Amendment 2).
- **"Cross-probe finding — three independent combos × three distinct
  parameter realizations all show the same overnight-session-structural
  signal" (verdict §Interpretation).** The pre-fix Probe 3 "3.45× overnight/
  RTH ratio on 865" was also built on the same TZ bug (see Probe 3
  Amendment 2). Under corrected TZ, 865 is RTH-leaning, 1298 is strongly
  RTH-concentrated, 664 is weakly overnight-leaning. The "signal family
  is overnight" framing was entirely a TZ artifact.

### What stands

- **Combo-1298 has a real, strong signal.** RTH-only Sharpe 4.414 on n=77
  is the cleanest single-combo readout in the project. It is still selected
  out of a 1500-combo sweep and tested on a partition consumed ≥ 5 times,
  so selection + partition-reuse discounts apply.
- **Probe 2's 1h test-partition single-gate PASS on combo-865** (Sharpe 2.895,
  220 trades) stands. It is session-agnostic.
- **Probe 1's family-level falsification direction** (N_1.3=4/1500 vs gate 10)
  stands. Engine-side session filter bug (see `memory/feedback_tz_source_ct.md`
  engine caveat) could shift which specific 4 combos passed, but direction
  is robust.

### Narrow-miss note

The narrow-miss flag is still False under corrected TZ (1298 abs PASS, not
[1.1, 1.3) miss). §5.2 is not invoked.

### Code fix

- `tasks/_probe4_readout.py:129`: `tz_localize("UTC")` →
  `tz_localize("America/Chicago", ambiguous="infer", nonexistent="shift_forward")`.
  Variable renamed `ts_utc` → `ts_ct`.
- Re-ran `_probe4_readout.py` locally (pure pandas, no engine re-run required)
  to produce corrected `data/ml/probe4/readout.json`.

### References

- `lessons.md` `2026-04-23 tz_bug_in_session_decomposition`
- `memory/feedback_tz_source_ct.md`
- `memory/project_tz_bug_cascade.md`
- `tasks/probe3_verdict.md` Amendment 2 (parallel retraction)
- `tasks/scope_d_brief.md` Amendment (parallel retraction)
- `tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer-paper-trade-readiness.md`
- `scripts/data_pipeline/update_bars_yfinance.py:37` — authoritative vendor TZ marker
