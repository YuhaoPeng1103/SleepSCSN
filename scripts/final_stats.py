"""Merge every final run, compute paired statistics, and write a report.

Sources
-------
``results/scsn_final`` and ``results/scsn_gn_manual``  SCSN matrix, one JSON
per (tag, model, fold, seed), each holding the arms of the run.
``results/adabn_final``  layerwise AdaBN and SCSN stacked on AdaBN, same runs.
``results/tent``  Tent baseline, same runs where it was run.

Output
------
``results/final_stats.json`` with per-dataset arm means and every paired
contrast used in the paper, plus a readable text report on stdout.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("results")
METRICS = ("accuracy", "kappa", "mf1", "ece", "coverage", "set_size",
           "review_workload", "worst_subject_coverage", "coverage_sd",
           "coverage_q10", "coverage_median", "probe", "probe_within_mean")

DATASETS = [
    ("edf_bn", "eegnet", "Sleep-EDF", "EEGNet"),
    ("edf_bn", "sleepcnn", "Sleep-EDF", "SleepCNN"),
    ("hmc_bn", "eegnet", "HMC", "EEGNet"),
    ("hmc_bn", "sleepcnn", "HMC", "SleepCNN"),
    ("edf_gn", "eegnet", "Sleep-EDF", "EEGNet"),
    ("edf_gn", "sleepcnn", "Sleep-EDF", "SleepCNN"),
    ("hmc_gn", "eegnet", "HMC", "EEGNet"),
    ("hmc_gn", "sleepcnn", "HMC", "SleepCNN"),
]


def with_within_mean(results: dict) -> dict:
    for arm in results.values():
        pw = arm.get("probe_within")
        if isinstance(pw, dict):
            vals = [v for v in pw.values() if v is not None]
            if vals:
                arm.setdefault("probe_within_mean", float(np.mean(vals)))
    return results


def load_scsn(tag: str, model: str) -> dict:
    runs = {}
    for folder in ("scsn_final", "scsn_gn_manual"):
        for path in glob.glob(str(ROOT / folder / f"{tag}_{model}_fold*_seed*.json")):
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            key = (int(d["fold"]), int(d["seed"]))
            d["results"] = with_within_mean(d["results"])
            runs[key] = d
    return runs


def load_adabn(tag: str, model: str) -> dict:
    runs = {}
    for path in glob.glob(str(ROOT / "adabn_final" / f"{tag}_{model}_fold*_seed*.json")):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        runs[(int(d["fold"]), int(d["seed"]))] = d
    return runs


def load_tent(cohort: str, model: str) -> dict:
    prefix = "" if cohort == "edf" else "hmc_"
    runs = {}
    for path in glob.glob(str(ROOT / "tent" / f"{prefix}{model}_fold*_seed*.json")):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        runs[(int(d["fold"]), int(d["seed"]))] = d
    return runs


def arm_means(runs: dict, arm: str) -> dict:
    entries = [r["results"][arm] for r in runs.values() if arm in r["results"]]
    if not entries:
        return {}
    out = {}
    for m in METRICS:
        vals = [e[m] for e in entries if e.get(m) is not None]
        if vals:
            out[m] = float(np.mean(vals))
    out["n"] = len(entries)
    return out


def paired(runs_a: dict, arm_a: str, runs_b: dict, arm_b: str) -> dict:
    keys = sorted(set(runs_a) & set(runs_b))
    diffs = {m: [] for m in METRICS}
    folds = defaultdict(list)
    used = []
    for key in keys:
        ra, rb = runs_a[key]["results"], runs_b[key]["results"]
        if arm_a not in ra or arm_b not in rb:
            continue
        used.append(key)
        for m in METRICS:
            a, b = ra[arm_a].get(m), rb[arm_b].get(m)
            if a is not None and b is not None:
                diffs[m].append(a - b)
        if ra[arm_a].get("accuracy") is not None and rb[arm_b].get("accuracy") is not None:
            folds[key[0]].append(ra[arm_a]["accuracy"] - rb[arm_b]["accuracy"])
    out = {"n": len(used), "arms": [arm_a, arm_b]}
    for m, d in diffs.items():
        if len(d) < 3:
            continue
        d = np.asarray(d, float)
        se = d.std(ddof=1) / np.sqrt(len(d))
        tcrit = stats.t.ppf(0.975, len(d) - 1)
        try:
            p = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            p = 1.0
        out[m] = {
            "diff": float(d.mean()),
            "ci": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
            "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
            "p": p,
            "signs": int((d > 0).sum()),
            "fold_signs": int(sum(np.nanmean(v) > 0 for v in folds.values())),
            "n_folds": len(folds),
        }
    return out


def main() -> None:
    out = {"datasets": {}, "pairs": {}}
    print(f"{'dataset':22s} {'runs':>4s}  arms")
    for tag, model, cohort, arch in DATASETS:
        scsn = load_scsn(tag, model)
        ada = load_adabn(tag, model)
        # Tent is a batch-normalisation method; on group-normalised checkpoints
        # it has no parameters to update, so it is not run there.
        tent = (load_tent("edf" if tag.startswith("edf") else "hmc", model)
                if tag.endswith("_bn") else {})
        merged = {}
        for key in sorted(set(scsn) | set(ada) | set(tent)):
            results = {}
            if key in scsn:
                results.update(scsn[key]["results"])
            if key in ada:
                results.update(ada[key]["results"])
            if key in tent:
                for arm, val in tent[key]["results"].items():
                    results.setdefault(arm, val)
            if results:
                merged[key] = {"model": model, "fold": key[0], "seed": key[1],
                               "results": with_within_mean(results)}
        arms = sorted(set().union(*(r["results"] for r in merged.values())))
        out["datasets"][f"{tag}_{model}"] = {
            "cohort": cohort, "arch": arch, "arms": {a: arm_means(merged, a) for a in arms},
            "n_runs": len(merged),
        }
        print(f"{tag + '_' + model:22s} {len(merged):4d}  {', '.join(arms)}")
        # Save merged runs for the table/figure scripts.
        (ROOT / "merged").mkdir(exist_ok=True)
        (ROOT / "merged" / f"{tag}_{model}.json").write_text(
            json.dumps({"runs": [{"fold": k[0], "seed": k[1], **v}
                                 for k, v in sorted(merged.items())]},
                       indent=1), encoding="utf-8")

        contrasts = [
            ("cond_full", "bn"), ("cond_full", "marginal"),
            ("cond_full", "adabn_layerwise"),
            ("cond_full", "cond_full_lam0.5"), ("cond_full", "cond_full_lam0.75"),
            ("cond_full", "cond_r1"), ("cond_full", "cond_r2"),
            ("cond_full", "cond_r4"), ("cond_full", "cond_r4_orth"),
            ("cond_full", "cond_hard"), ("cond_full", "cond_mismatch"),
            ("cond_full", "tent_sgd"), ("cond_full", "tent1"),
            ("cond_full", "tent5"),
            ("adabn_layerwise", "bn"), ("adabn_layerwise", "marginal"),
            ("ada_scsn", "adabn_layerwise"),
            ("marginal", "bn"),
        ]
        for a, b in contrasts:
            if a == b:
                continue
            p = paired(merged, a, merged, b)
            if p.get("n", 0) >= 3:
                out["pairs"][f"{tag}_{model}:{a}_vs_{b}"] = p

    (ROOT / "final_stats.json").write_text(json.dumps(out, indent=1),
                                           encoding="utf-8")
    print("\nwrote results/final_stats.json")

    print("\n=== head-line means ===")
    for key, ds in out["datasets"].items():
        print(f"\n{key}  n={ds['n_runs']}")
        for arm in ("bn", "marginal", "adabn_layerwise", "cond_full",
                    "cond_full_lam0.5", "cond_r4", "cond_r4_orth",
                    "cond_hard", "cond_mismatch", "tent_sgd"):
            a = ds["arms"].get(arm)
            if not a:
                continue
            print(f"  {arm:18s} n={a.get('n'):2d} "
                  f"acc={a.get('accuracy', float('nan')):.4f} "
                  f"worst={a.get('worst_subject_coverage', float('nan')):.4f} "
                  f"sd={a.get('coverage_sd', float('nan')):.4f} "
                  f"probe={a.get('probe', float('nan')):.4f} "
                  f"within={a.get('probe_within_mean', float('nan')):.4f}")

    print("\n=== key contrasts (accuracy / within-probe) ===")
    for key, p in out["pairs"].items():
        acc, pro = p.get("accuracy"), p.get("probe_within_mean")
        if acc is None and pro is None:
            continue
        bits = []
        if acc:
            bits.append(f"acc {acc['diff']:+.4f} [{acc['ci'][0]:+.4f},"
                        f"{acc['ci'][1]:+.4f}] p={acc['p']:.4g}")
        if pro:
            bits.append(f"within {pro['diff']:+.4f} [{pro['ci'][0]:+.4f},"
                        f"{pro['ci'][1]:+.4f}] p={pro['p']:.4g}")
        print(f"{key:60s} n={p['n']:2d}  " + "  ".join(bits))


if __name__ == "__main__":
    main()
