"""Summarise the lambda_s sensitivity sweep.

The main runs use lambda_s = 10 (``results/scsn_final``); the sweep reruns the
SCSN arm with lambda_s in {1, 100} on the same checkpoints and the same cached
features (``results/scsn_lams``).  This script reports, per setting, the mean
accuracy and within-stage probe at each value, so the manuscript can state how
sensitive the result is to the single hyperparameter.

Usage:
    python scripts/lam_s_summary.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [
    ("edf_bn_eegnet", "edf_bn_eegnet"),
    ("edf_bn_sleepcnn", "edf_bn_sleepcnn"),
    ("hmc_bn_eegnet", "hmc_bn_eegnet"),
    ("hmc_bn_sleepcnn", "hmc_bn_sleepcnn"),
]
LAMS = (1.0, 10.0, 100.0)


def within_mean(arm: dict) -> float | None:
    pw = arm.get("probe_within")
    if not isinstance(pw, dict):
        return None
    vals = [v for v in pw.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def collect(tag: str, lam: float) -> dict[str, list[float]]:
    if lam == 10.0:
        paths = sorted(glob.glob(str(ROOT / "results/scsn_final" / f"{tag}_fold*_seed*.json")))
    else:
        paths = sorted(glob.glob(str(ROOT / "results/scsn_lams" / f"{tag}_fold*_seed*_lams{lam:g}.json")))
    acc, within, probe, worst, sd = [], [], [], [], []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        arm = d["results"]["cond_full"]
        acc.append(arm["accuracy"])
        w = within_mean(arm)
        if w is not None:
            within.append(w)
        probe.append(arm["probe"])
        worst.append(arm["worst_subject_coverage"])
        sd.append(arm["coverage_sd"])
    return {
        "n": len(acc),
        "accuracy": acc,
        "within": within,
        "probe": probe,
        "worst": worst,
        "sd": sd,
    }


def main() -> None:
    out = {}
    for tag, _ in SETTINGS:
        out[tag] = {}
        for lam in LAMS:
            runs = collect(tag, lam)
            out[tag][f"{lam:g}"] = {
                "n": runs["n"],
                "accuracy": float(np.mean(runs["accuracy"])),
                "within": float(np.mean(runs["within"])),
                "probe": float(np.mean(runs["probe"])),
                "worst": float(np.mean(runs["worst"])),
                "sd": float(np.mean(runs["sd"])),
            }
    print(f"{'setting':22s} {'lam_s':>5s} {'n':>3s} {'acc':>7s} {'within':>7s} {'probe':>7s} {'worst':>7s} {'sd':>6s}")
    for tag, _ in SETTINGS:
        for lam in LAMS:
            r = out[tag][f"{lam:g}"]
            print(f"{tag:22s} {lam:5.0f} {r['n']:3d} {r['accuracy']:7.4f} {r['within']:7.4f} "
                  f"{r['probe']:7.4f} {r['worst']:7.4f} {r['sd']:6.4f}")
    (ROOT / "results/lam_s_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
