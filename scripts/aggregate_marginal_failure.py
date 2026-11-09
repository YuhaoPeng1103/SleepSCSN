"""Aggregate the per-recording marginal-failure analyses over HMC runs."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    files = sorted(glob.glob("results/pilot_cssn2/marginal_failure_*.json"))
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in files]
    rows = [r for r in rows if "spearman_js_vs_delta_marginal" in r]
    if not rows:
        raise SystemExit("no marginal-failure files")
    for key in ("spearman_js_vs_delta_marginal", "spearman_p_marginal",
                "spearman_js_vs_delta_conditional", "spearman_p_conditional",
                "mean_delta_marginal", "mean_delta_conditional"):
        v = np.asarray([r[key] for r in rows], dtype=float)
        print(f"{key:40s} mean={v.mean():+.4f} sd={v.std(ddof=1):.4f} n={v.size}")
    neg = sum(r["spearman_js_vs_delta_marginal"] < 0 for r in rows)
    print(f"marginal rho < 0 in {neg}/{len(rows)} runs")
    out = Path("results/pilot_cssn2/marginal_failure_summary.json")
    out.write_text(json.dumps({
        "n_runs": len(rows),
        "rho_marginal_mean": float(np.mean(
            [r["spearman_js_vs_delta_marginal"] for r in rows])),
        "rho_conditional_mean": float(np.mean(
            [r["spearman_js_vs_delta_conditional"] for r in rows])),
        "delta_marginal_mean": float(np.mean(
            [r["mean_delta_marginal"] for r in rows])),
        "delta_conditional_mean": float(np.mean(
            [r["mean_delta_conditional"] for r in rows])),
        "runs": [str(Path(p).name) for p in files],
    }, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
