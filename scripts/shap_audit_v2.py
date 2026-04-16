"""
B8-SHAP audit — is Family A's lift time-local signal or combo-ID leakage?

Trains a Family-A LightGBM on fold-0 of the B7 testbed, then uses
TreeSHAP (`pred_contrib=True`) on a sample of the fold-0 validation rows to
diagnose whether `prior_wr_50` (and siblings) are carrying within-combo
dynamics or merely proxying each combo's baseline WR.

Diagnostics emitted:
  - Mean |SHAP| summary bar (Family A feature importance vs V2 features).
  - `prior_wr_50` dependence scatter coloured by combo_id (20 combos).
  - Per-combo SHAP boxplot for `prior_wr_50` (top-20 combos by trade count).
  - JSON: within_combo_std / between_combo_std ratio per Family-A feature.
     Ratio >= 1.0 → dynamic signal. < 0.3 → combo-ID proxy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from feature_engineering_v2 import (  # noqa: E402
    CATEGORICAL_COLS,
    FAMILY_A,
    LGB_PARAMS,
    PARQUET_COLUMNS,
    RR_FEATURE,
    V2_COMBO_FEATURES,
    V2_DERIVED_FEATURES,
    V2_ENTRY_FEATURES,
    add_family_a,
    expand,
)

TESTBED = REPO_ROOT / "data/ml/mfe/ml_dataset_v10_train_wf_mfe.parquet"
OUT_DIR = REPO_ROOT / "data/ml/adaptive_rr_v2"
MAX_BASE_TRADES = 200_000   # keeps local memory < 2 GB
SHAP_SAMPLE = 100_000       # SHAP-evaluation rows (expanded)
SEED = 42


def load_base(n_cap: int) -> pd.DataFrame:
    pf = pq.ParquetFile(TESTBED)
    have = {f.name for f in pf.schema_arrow}
    cols = [c for c in PARQUET_COLUMNS if c in have]
    df = pf.read(columns=cols).to_pandas()
    df = df.dropna(subset=["mfe_points", "stop_distance_pts"]).reset_index(drop=True)
    print(f"[shap] loaded {len(df):,} base trades")
    if len(df) > n_cap:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(df), size=n_cap, replace=False)
        df = df.iloc[sorted(keep)].reset_index(drop=True)
        print(f"[shap] subsampled to {len(df):,} for local memory")
    df = df.sort_values(["combo_id", "entry_bar_idx"]).reset_index(drop=True)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df_base = load_base(MAX_BASE_TRADES)
    df_base = add_family_a(df_base)
    print(f"[shap] Family-A features added: {FAMILY_A}")

    ex = expand(df_base, FAMILY_A)
    feat_cols = (V2_ENTRY_FEATURES + V2_COMBO_FEATURES
                 + [RR_FEATURE] + V2_DERIVED_FEATURES + FAMILY_A)
    feat_cols = [c for c in feat_cols if c in ex.columns]
    X = ex[feat_cols]
    y = ex["would_win"].to_numpy(dtype=np.int8)
    groups = ex["__combo_id__"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(iter(sgkf.split(X, y, groups)))
    print(f"[shap] fold-0 train={len(tr_idx):,} val={len(va_idx):,}")

    lgb_train = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                            categorical_feature=CATEGORICAL_COLS,
                            free_raw_data=True)
    print(f"[shap] training LightGBM ({LGB_PARAMS['n_estimators']} rounds)...")
    t_train = time.time()
    model = lgb.train(LGB_PARAMS, lgb_train,
                      num_boost_round=LGB_PARAMS["n_estimators"])
    print(f"[shap] train done in {time.time()-t_train:.0f}s")

    # SHAP on a random subsample of the fold-0 validation rows.
    rng = np.random.default_rng(SEED)
    if len(va_idx) > SHAP_SAMPLE:
        pick = rng.choice(va_idx, size=SHAP_SAMPLE, replace=False)
    else:
        pick = va_idx
    X_shap = X.iloc[pick]
    combos_shap = groups[pick]

    t_shap = time.time()
    contrib = model.predict(X_shap, pred_contrib=True, num_threads=4)
    contrib = np.asarray(contrib, dtype=np.float32)  # (n_rows, n_features+1)
    shap_vals = contrib[:, :-1]
    print(f"[shap] SHAP computed in {time.time()-t_shap:.0f}s "
          f"shape={shap_vals.shape}")

    # ---- Diagnostic 1: mean |SHAP| bar chart ---------------------------
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    feat_names_ord = [feat_cols[i] for i in order]
    colors = ["tab:orange" if f in FAMILY_A else "tab:blue"
              for f in feat_names_ord]
    ax.barh(range(len(feat_cols)), mean_abs[order], color=colors)
    ax.set_yticks(range(len(feat_cols)))
    ax.set_yticklabels(feat_names_ord, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Family A (orange) vs V2 (blue) — feature importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shap_summary_v2.png", dpi=140)
    plt.close(fig)
    print("[shap] saved shap_summary_v2.png")

    # ---- Diagnostic 2: prior_wr_50 dependence scatter ------------------
    pwr50_idx = feat_cols.index("prior_wr_50")
    pwr50_vals = X_shap["prior_wr_50"].to_numpy()
    pwr50_shap = shap_vals[:, pwr50_idx]

    # Pick 20 combos with most rows in this SHAP sample for colouring.
    unique_combos, counts = np.unique(combos_shap, return_counts=True)
    top20 = unique_combos[np.argsort(counts)[::-1][:20]]
    mask20 = np.isin(combos_shap, top20)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20", len(top20))
    for i, c in enumerate(top20):
        sel = combos_shap[mask20] == c
        ax.scatter(pwr50_vals[mask20][sel], pwr50_shap[mask20][sel],
                   s=4, alpha=0.4, color=cmap(i), label=f"combo {int(c)}")
    ax.set_xlabel("prior_wr_50 value")
    ax.set_ylabel("SHAP contribution to log-odds")
    ax.set_title("prior_wr_50 dependence (coloured by combo) — "
                 "tight per-combo clusters → identity proxy, "
                 "spread → dynamic signal")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shap_prior_wr_50_dependence_v2.png", dpi=140)
    plt.close(fig)
    print("[shap] saved shap_prior_wr_50_dependence_v2.png")

    # ---- Diagnostic 3: within-combo boxplot of prior_wr_50 SHAP -------
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [pwr50_shap[combos_shap == c] for c in top20]
    ax.boxplot(data, labels=[str(int(c)) for c in top20],
               showfliers=False)
    ax.set_xlabel("combo_id (top 20 by sample count)")
    ax.set_ylabel("prior_wr_50 SHAP")
    ax.set_title("Per-combo SHAP spread — wide boxes = dynamic signal, "
                 "narrow = identity proxy")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shap_per_combo_boxplot_v2.png", dpi=140)
    plt.close(fig)
    print("[shap] saved shap_per_combo_boxplot_v2.png")

    # ---- Diagnostic 4: within/between std ratio per Family-A feat ------
    summary = {"n_shap_rows": int(len(pick)),
               "n_combos_in_sample": int(len(unique_combos)),
               "feature_ratios": {}}
    for feat in FAMILY_A:
        if feat not in feat_cols:
            continue
        fi = feat_cols.index(feat)
        sv = shap_vals[:, fi]
        # Per-combo means.
        combo_means = {}
        for c in unique_combos:
            sel = combos_shap == c
            if sel.sum() >= 50:
                combo_means[int(c)] = float(sv[sel].mean())
        between_std = float(np.std(list(combo_means.values())))
        # Within-combo std: pool residuals.
        residuals = []
        for c, mean in combo_means.items():
            sel = combos_shap == c
            residuals.extend((sv[sel] - mean).tolist())
        within_std = float(np.std(residuals)) if residuals else 0.0
        ratio = within_std / between_std if between_std > 1e-9 else float("inf")
        summary["feature_ratios"][feat] = {
            "within_std": within_std,
            "between_std": between_std,
            "ratio": ratio,
            "verdict": ("dynamic" if ratio >= 1.0
                        else "partial" if ratio >= 0.3
                        else "identity_proxy"),
            "mean_abs_shap": float(np.mean(np.abs(sv))),
        }

    summary["runtime_seconds"] = time.time() - t0
    summary["n_train_rows"] = int(len(tr_idx))
    summary["n_val_rows"] = int(len(va_idx))
    summary["testbed"] = str(TESTBED)
    out_path = OUT_DIR / "shap_audit_v2.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[shap] saved {out_path}")

    print("\n[shap] Family-A verdicts (within/between ratio):")
    for feat, d in summary["feature_ratios"].items():
        print(f"  {feat:18s}  ratio={d['ratio']:.3f}  "
              f"within={d['within_std']:.4f}  between={d['between_std']:.4f}  "
              f"-> {d['verdict']}")


if __name__ == "__main__":
    main()
