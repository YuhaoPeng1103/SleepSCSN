"""Assemble the replication on the external HMC cohort.

Same three arms as ``aggregate_arms.py`` -- ``batch``, ``group`` and ``adabn``
-- but on ``results/hmc``. The keys are ``eegnet/fold<f>_seed<s>``, so the
paired tests again compare runs trained on the same subjects with the same
initialisation.

Example
-------
python scripts/aggregate_hmc.py --out results/norm/arms_hmc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ``python -I`` drops the script's own directory from sys.path, so the sibling
# module has to be added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_arms import (METRICS, collect, collect_adabn, collect_probes,
                            paired)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/hmc"))
    parser.add_argument("--adabn", type=Path,
                        default=Path("results/norm/adabn_hmc_eegnet.json"))
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--out", type=Path, default=Path("results/norm/arms_hmc.json"))
    args = parser.parse_args()

    arms = {
        "batch": collect(args.root / "batch"),
        "group": collect(args.root / "group"),
    }
    for arm in ("batch", "group"):
        summary_dir = args.root / arm / "summary"
        for key, value in collect_probes(summary_dir).items():
            if key in arms[arm]:
                arms[arm][key].update(value)

    arms["adabn"] = collect_adabn(args.adabn) if args.adabn.exists() else {}

    # Keep only the requested backbone: the results tree holds one directory
    # per model, and a paired test across models would compare runs that were
    # never trained on the same initialisation.
    prefix = f"{args.model}/"
    for arm in ("batch", "group", "adabn"):
        arms[arm] = {k: v for k, v in arms.get(arm, {}).items()
                     if k.startswith(prefix)}

    result: dict = {"arms": arms, "pairs": {}, "summary": {}}
    keys = sorted(arms["batch"])
    for a, b in (("batch", "group"), ("batch", "adabn"), ("group", "adabn")):
        common = [k for k in keys if k in arms[a] and k in arms[b]]
        if len(common) >= 3:
            result["pairs"][f"{args.model}|{a}-vs-{b}"] = paired(arms[a], arms[b], common)

    for name, arm in arms.items():
        if not arm:
            continue
        rows = list(arm.values())
        # Nested under the backbone so that the same ``main_table`` used for
        # the Sleep-EDF table can render this one.
        result["summary"][name] = {args.model: {
            metric: float(np.mean([r[metric] for r in rows if r.get(metric) is not None]))
            for metric in METRICS
            if any(r.get(metric) is not None for r in rows)
        }}
        result["summary"][name][args.model]["n_runs"] = len(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    print(f"{'comparison':22s} {'metric':14s} {'a':>7s} {'b':>7s} {'diff':>8s} "
          f"{'dz':>6s} {'p':>9s} {'n':>3s}")
    for name, stats_ in result["pairs"].items():
        for metric in ("accuracy", "set_size", "worst", "coverage_sd", "probe"):
            if metric in stats_:
                s = stats_[metric]
                print(f"{name:22s} {metric:14s} {s['a_mean']:7.3f} {s['b_mean']:7.3f} "
                      f"{s['diff']:+8.3f} {s['dz']:+6.2f} {s['wilcoxon_p']:9.4f} {s['n']:3d}")
    for name, by_model in result["summary"].items():
        for model, stats_ in by_model.items():
            print(name, model, {k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in stats_.items()
                                if k in ("accuracy", "coverage", "set_size",
                                         "worst", "coverage_sd", "probe",
                                         "n_runs")})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
