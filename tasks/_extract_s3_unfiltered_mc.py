"""Extract unfiltered Pool B Monte Carlo verdict from s3_mc_combined_net.ipynb.

Kill-criterion input: S_unfiltered_C (p50 Sharpe) and R_unfiltered_C (ruin prob)
under V3-no-ID training, no ML#2 filter, fixed_dollars_500, $5/contract RT.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

NB_PATH = Path("evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s3_mc_combined_net.ipynb")


def main():
    if not NB_PATH.exists():
        raise SystemExit(f"[ERR] notebook not found at {NB_PATH}")

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    print(f"[notebook] {NB_PATH.name}  cells={len(cells)}")

    hits = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not any(k in src for k in ("monte_carlo", "plot_mc_sharpe", "plot_mc_dd",
                                      "plot_mc_pnl", "sharpe_p50", "risk_of_ruin",
                                      "ruin_prob", "sharpe_ci")):
            continue
        outs = cell.get("outputs", [])
        out_text_parts = []
        for o in outs:
            t = o.get("output_type")
            if t == "stream":
                out_text_parts.append("".join(o.get("text", [])))
            elif t in ("execute_result", "display_data"):
                data = o.get("data", {}) or {}
                if "text/plain" in data:
                    out_text_parts.append("".join(data["text/plain"]))
        out_text = "\n".join(out_text_parts)
        if out_text.strip():
            hits.append((idx, src[:220], out_text))

    print(f"[hits] {len(hits)} MC-relevant cells with non-empty output")
    print()

    for idx, src_head, out_text in hits:
        print("=" * 80)
        print(f"[cell {idx}] source head:")
        print(src_head)
        print("-" * 80)
        print("[output]:")
        print(out_text[:3500])
        print()


if __name__ == "__main__":
    main()
