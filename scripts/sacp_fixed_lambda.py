"""Warm-up calibration at a *fixed* shrinkage weight.

``run_sacp.py`` selects one ``lambda`` per (run, budget) by leave-one-subject-out
simulation on the calibration subjects and reports only the selected value.
That leaves a question a referee is entitled to ask: how much of the SACP gain
comes from the shrinkage at all, and would the simplest deployable rule ---
``lambda = 1``, i.e. pure per-subject recalibration with no shrinkage --- do
just as well?

This script answers it from the same fitted models, without retraining. For
every run directory it rebuilds the score matrices exactly as ``run_sacp.py``
does and evaluates the warm-up rule at every candidate ``lambda`` for every
budget, so the selected operating point can be compared against the fixed
alternatives on the test set.

Usage::

    python scripts/sacp_fixed_lambda.py --root results/main \
        --out results/main/summary/sacp_fixed_lambda.json

The output is a table of coverage / worst-patient coverage / per-patient SD /
set size, averaged over the runs, indexed by (budget, lambda).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import (
    aps_score_matrix,
    raps_score_matrix,
    shrink_quantile,
    split_conformal_quantile,
    tune_raps,
)
from sleepcp.temporal import initial_distribution, smooth_grouped, transition_matrix


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def subject_slices(subjects: np.ndarray) -> dict[str, np.ndarray]:
    return {str(s): np.flatnonzero(subjects == s) for s in np.unique(subjects)}


def split_first(idx: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    if m <= 0:
        return idx[:0], idx
    m = min(m, idx.size)
    return idx[:m], idx[m:]


def summarise(sets: np.ndarray, labels: np.ndarray, subjects: np.ndarray,
              alpha: float) -> dict:
    per_subject = {}
    for s in np.unique(subjects):
        mask = subjects == s
        per_subject[str(s)] = float(sets[mask, labels[mask]].mean())
    values = np.asarray(list(per_subject.values()), dtype=float)
    return {
        "coverage": float(sets[np.arange(labels.size), labels].mean()),
        "set_size": float(sets.sum(axis=1).mean()),
        "per_subject_coverage": per_subject,
        "mean_subject_coverage": float(values.mean()),
        "worst_subject_coverage": float(values.min()),
        "std_subject_coverage": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "subject_violation_rate": float((values < 1.0 - alpha).mean()),
        "n_epochs": int(labels.size),
    }


def run_one(cal_path: Path, test_path: Path, alpha: float, budgets: list[int],
            lambdas: list[float], score: str, smooth: bool) -> dict:
    cal_raw = load_npz(cal_path)
    test_raw = load_npz(test_path)
    cal_labels = cal_raw["labels"].astype(int)
    test_labels = test_raw["labels"].astype(int)
    cal_subjects = cal_raw["subjects"].astype(str)
    test_subjects = test_raw["subjects"].astype(str)

    cal_probs = cal_raw["probs"].astype(float)
    test_probs = test_raw["probs"].astype(float)
    if smooth:
        trans = transition_matrix(cal_labels, smoothing=1.0)
        init = initial_distribution(cal_labels)
        cal_probs = smooth_grouped(
            cal_probs, cal_raw["recordings"].astype(str), trans, init
        )
        test_probs = smooth_grouped(
            test_probs, test_raw["recordings"].astype(str), trans, init
        )

    if score == "raps":
        k_reg, raps_lam = tune_raps(cal_probs, cal_labels, alpha)
        cal_scores = raps_score_matrix(cal_probs, k_reg=k_reg, lam=raps_lam)
        test_scores = raps_score_matrix(test_probs, k_reg=k_reg, lam=raps_lam)
    else:
        cal_scores = aps_score_matrix(cal_probs)
        test_scores = aps_score_matrix(test_probs)

    cal_true = cal_scores[np.arange(cal_labels.size), cal_labels]
    global_q = split_conformal_quantile(cal_true, alpha)
    test_sl = subject_slices(test_subjects)

    out = {}
    for m in budgets:
        # The local threshold depends on the subject's own warm-up only, so it
        # is computed once per (subject, budget) and reused across lambdas.
        local = {}
        for s, idx in test_sl.items():
            warm, eval_idx = split_first(idx, m)
            q = float("inf")
            if warm.size:
                q = split_conformal_quantile(test_scores[warm, test_labels[warm]], alpha)
            local[s] = (q, eval_idx)

        for lam in lambdas:
            sets_all, labels_all, subjects_all = [], [], []
            n_fallback = 0
            for s, (local_q, eval_idx) in local.items():
                if eval_idx.size == 0:
                    continue
                if not np.isfinite(local_q):
                    n_fallback += 1
                    q = global_q
                elif m == 0:
                    q = global_q
                else:
                    q = shrink_quantile(local_q, global_q, lam)
                sets_all.append(test_scores[eval_idx] <= q)
                labels_all.append(test_labels[eval_idx])
                subjects_all.append(np.repeat(s, eval_idx.size))
            rec = summarise(
                np.concatenate(sets_all),
                np.concatenate(labels_all),
                np.concatenate(subjects_all),
                alpha,
            )
            rec["subjects_needing_global_fallback"] = n_fallback
            out[f"m{m}_lam{lam:g}"] = rec
    return out


def tally_selection(root: Path, models: list[str], budgets: list[int]) -> dict:
    """How often LOSO selection picks each weight, read from ``sacp.json``."""
    hist = {str(m): {} for m in budgets}
    all_zero_upto_80 = 0
    n_runs = 0
    for model in models:
        for run_dir in sorted((root / model).glob("fold*_seed*")):
            path = run_dir / "sacp.json"
            if not path.exists():
                continue
            curve = json.loads(path.read_text(encoding="utf-8"))["budget_curve"]
            n_runs += 1
            for m in budgets:
                key = str(m)
                if key not in curve:
                    continue
                lam = round(float(curve[key]["lambda"]), 2)
                hist[key][lam] = hist[key].get(lam, 0) + 1
            small = [round(float(curve[str(m)]["lambda"]), 2)
                     for m in budgets if m <= 80 and str(m) in curve]
            if small and all(v == 0.0 for v in small):
                all_zero_upto_80 += 1
    return {"n_runs": n_runs, "hist_per_budget": hist,
            "all_zero_up_to_80": all_zero_upto_80}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="matrix root, e.g. results/main")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["eegnet", "sleepcnn"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--budget", type=int, nargs="+", default=[20, 80, 160, 320])
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--score", default="aps", choices=["aps", "raps"])
    parser.add_argument("--smooth", action="store_true")
    args = parser.parse_args()

    per_run = {}
    for model in args.models:
        for run_dir in sorted((args.root / model).glob("fold*_seed*")):
            cal = next(run_dir.glob("*_cal.npz"), None)
            test = next(run_dir.glob("*_test.npz"), None)
            if cal is None or test is None:
                print(f"skip {run_dir}: missing npz")
                continue
            tag = f"{model}/{run_dir.name}"
            print(f"== {tag} ==", flush=True)
            per_run[tag] = run_one(cal, test, args.alpha, args.budget,
                                   args.lambdas, args.score, args.smooth)

    keys = sorted(next(iter(per_run.values())).keys()) if per_run else []
    table = {}
    for key in keys:
        vals = [r[key] for r in per_run.values() if key in r]
        if not vals:
            continue
        # Per-subject coverage is merged across folds, exactly as in
        # ``scripts/aggregate.py``: one value per subject, averaged over the
        # runs that contain it, so that "worst subject" here is the same
        # statistic as in the other tables (not the worst of four per fold,
        # and not whichever fold happened to be read last).
        total: dict[str, float] = {}
        count: dict[str, int] = {}
        for rec in vals:
            for subj, value in rec["per_subject_coverage"].items():
                total[subj] = total.get(subj, 0.0) + float(value)
                count[subj] = count.get(subj, 0) + 1
        merged = {subj: total[subj] / count[subj] for subj in total}
        values = np.asarray(list(merged.values()), dtype=float)
        cov = np.asarray([v["coverage"] for v in vals], dtype=float)
        size = np.asarray([v["set_size"] for v in vals], dtype=float)
        table[key] = {
            "n_subjects": int(values.size),
            "coverage": {
                "mean": float(cov.mean()),
                "sd": float(cov.std(ddof=1)) if cov.size > 1 else 0.0,
            },
            "set_size": {
                "mean": float(size.mean()),
                "sd": float(size.std(ddof=1)) if size.size > 1 else 0.0,
            },
            "worst_subject_coverage": {"mean": float(values.min()), "sd": 0.0},
            "std_subject_coverage": {
                "mean": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "sd": 0.0,
            },
            "subject_violation_rate": {
                "mean": float((values < 1.0 - args.alpha).mean()), "sd": 0.0,
            },
        }

    out = {
        "alpha": args.alpha,
        "score": args.score,
        "smooth": bool(args.smooth),
        "budgets": args.budget,
        "lambdas": args.lambdas,
        "n_runs": len(per_run),
        "table": table,
        "losos_selected_lambda": tally_selection(args.root, args.models, args.budget),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'budget':>7} {'lambda':>6} {'coverage':>9} {'worst':>7} "
          f"{'sd':>6} {'size':>6}")
    for m in args.budget:
        for lam in args.lambdas:
            key = f"m{m}_lam{lam:g}"
            if key not in table:
                continue
            r = table[key]
            print(f"{m:>7} {lam:>6.2f} {r['coverage']['mean']:>9.3f} "
                  f"{r['worst_subject_coverage']['mean']:>7.3f} "
                  f"{r['std_subject_coverage']['mean']:>6.3f} "
                  f"{r['set_size']['mean']:>6.3f}")
    sel = out["losos_selected_lambda"]
    print(f"\nLOSO-selected lambda over {sel['n_runs']} runs "
          f"(budgets <= 80 all zero in {sel['all_zero_up_to_80']}):")
    for m in args.budget:
        hist = sel["hist_per_budget"].get(str(m), {})
        if hist:
            print(f"  m={m:>3}: " + ", ".join(f"{k:g}x{v}" for k, v in sorted(hist.items())))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
