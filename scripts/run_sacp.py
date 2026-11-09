"""Subject-adaptive conformal prediction (SACP) for sleep staging.

The script studies the coverage / efficiency / labelling-budget trade-off of
three calibration regimes when calibration and test subjects are disjoint:

``global``
    One threshold for every subject, calibrated on the pooled calibration
    subjects.

``subject``
    A per-subject threshold calibrated on the first ``m`` labelled epochs of
    that subject (a warm-up), shrunk toward the global threshold:

        q_i(lambda) = (1 - lambda) q_global + lambda q_i.

``neighborhood``
    No labels from the test subject. Subjects are matched on unlabelled
    descriptors (mean predicted class probabilities, mean confidence, mean
    entropy), the k nearest calibration subjects provide a local threshold,
    and that threshold is shrunk toward the global one.

``oracle``
    Per-subject threshold calibrated on all of that subject's test epochs.
    Not deployable; it is the best a subject-specific threshold could do.

Warm-up epochs are removed from the evaluation set, so coverage is never
measured on data used for calibration.  ``lambda`` is selected by
leave-one-subject-out simulation on the calibration subjects, not tuned on the
test set.

Outputs a JSON with a coverage-vs-budget curve and per-method summaries.
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
from sleepcp.metrics import (
    coverage,
    empty_rate,
    per_class_coverage,
    review_workload,
    set_size,
    singleton_rate,
)
from sleepcp.temporal import initial_distribution, smooth_grouped, transition_matrix


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def subject_features(probs: np.ndarray) -> np.ndarray:
    mean_probs = probs.mean(axis=0)
    mean_max = probs.max(axis=1).mean()
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return np.concatenate([mean_probs, [mean_max, float(entropy.mean())]])


def summarize(
    sets: np.ndarray, labels: np.ndarray, subjects: np.ndarray, alpha: float = 0.1
) -> dict:
    per_subject = {}
    for s in np.unique(subjects):
        mask = subjects == s
        per_subject[str(s)] = coverage(sets[mask], labels[mask])
    values = np.asarray(list(per_subject.values()), dtype=float)
    return {
        "coverage": coverage(sets, labels),
        "per_class_coverage": per_class_coverage(sets, labels).tolist(),
        "set_size": set_size(sets),
        "singleton_rate": singleton_rate(sets),
        "empty_rate": empty_rate(sets),
        "review_workload": review_workload(sets),
        "n_epochs": int(labels.size),
        "per_subject_coverage": per_subject,
        "mean_subject_coverage": float(values.mean()),
        "worst_subject_coverage": float(values.min()),
        "p10_subject_coverage": float(np.quantile(values, 0.1)),
        "p25_subject_coverage": float(np.quantile(values, 0.25)),
        "std_subject_coverage": float(values.std()),
        "subject_violation_rate": float((values < 1.0 - alpha).mean()),
    }


def subject_slices(subjects: np.ndarray) -> dict[str, np.ndarray]:
    """Indices of each subject's epochs, in stored (temporal) order."""
    return {str(s): np.flatnonzero(subjects == s) for s in np.unique(subjects)}


def split_warmup(
    idx: np.ndarray, m: int, mode: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split a subject's epochs into a warm-up and an evaluation part."""
    if m <= 0:
        return idx[:0], idx
    m = min(m, idx.size)
    if mode == "first":
        return idx[:m], idx[m:]
    perm = rng.permutation(idx.size)
    return idx[perm[:m]], idx[perm[m:]]


def choose_lambda_loso(
    cal_matrix: np.ndarray,
    cal_labels: np.ndarray,
    cal_subjects: np.ndarray,
    global_q: float,
    m: int,
    alpha: float,
    grid: np.ndarray,
    warmup_select: str = "first",
) -> tuple[float, dict]:
    """Pick lambda by leave-one-subject-out simulation on calibration subjects.

    Each calibration subject plays the role of a new subject: it gets a warm-up
    of ``m`` labelled epochs, the remaining calibration subjects form the pool,
    and the shrunk threshold is scored on the rest of that subject's epochs.
    The criterion is mean per-subject coverage violation, with set size as a
    tie-breaker.
    """
    sl = subject_slices(cal_subjects)
    subjects = list(sl)
    true_all = cal_matrix[np.arange(cal_matrix.shape[0]), cal_labels]

    records = {lam: {"cov": [], "size": []} for lam in grid}
    rng = np.random.default_rng(12345)
    for held in sorted(subjects):
        idx = sl[held]
        if idx.size <= m + 20:
            continue
        warm, rest = split_warmup(idx, m, warmup_select, rng)
        local_q = split_conformal_quantile(true_all[warm], alpha)
        for lam in grid:
            q = (1.0 - lam) * global_q + lam * local_q if m > 0 else global_q
            sets = cal_matrix[rest] <= q
            records[lam]["cov"].append(coverage(sets, cal_labels[rest]))

    # One-sided criterion: the average coverage shortfall below 1 - alpha.
    # Over-coverage is not rewarded beyond avoiding a shortfall, because the
    # cost of over-coverage shows up as larger sets, not as a coverage gain.
    best_lam, best_score = 0.0, np.inf
    detail = {}
    for lam, rec in records.items():
        if not rec["cov"]:
            continue
        shortfall = float(np.maximum(0.0, (1.0 - alpha) - np.asarray(rec["cov"])).mean())
        detail[str(lam)] = {"mean_coverage": float(np.mean(rec["cov"])), "shortfall": shortfall}
        if shortfall < best_score - 1e-12:
            best_score, best_lam = shortfall, float(lam)
    return best_lam, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--budget", type=int, nargs="+", default=[0, 10, 20, 40, 80, 160, 320])
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--cal-subjects",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12],
        help="subset sizes (in subjects) for the calibration-size curve",
    )
    parser.add_argument("--cal-repeats", type=int, default=20)
    parser.add_argument("--score", default="aps", choices=["aps", "raps"])
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="score the forward-backward posterior instead of the raw softmax",
    )
    parser.add_argument("--transition-smoothing", type=float, default=1.0)
    parser.add_argument(
        "--warmup-select",
        default="first",
        choices=["first", "random"],
        help="'first' uses the first m epochs of the night (realistic, exposed "
        "to within-night drift); 'random' draws m epochs at random and is only "
        "valid if the night is stationary",
    )
    parser.add_argument("--aci", action="store_true", help="also run adaptive conformal inference")
    parser.add_argument("--aci-gamma", type=float, default=0.05)
    args = parser.parse_args()

    cal_raw = load_npz(args.cal)
    test_raw = load_npz(args.test)
    cal = {
        "probs": cal_raw["probs"].astype(float),
        "labels": cal_raw["labels"].astype(int),
        "subjects": cal_raw["subjects"].astype(str),
    }
    test = {
        "probs": test_raw["probs"].astype(float),
        "labels": test_raw["labels"].astype(int),
        "subjects": test_raw["subjects"].astype(str),
    }
    if args.smooth:
        cal_rec = cal_raw["recordings"].astype(str)
        test_rec = test_raw["recordings"].astype(str)
        trans = transition_matrix(
            cal["labels"], smoothing=args.transition_smoothing
        )
        init = initial_distribution(cal["labels"])
        cal_probs = smooth_grouped(cal["probs"], cal_rec, trans, init)
        test_probs = smooth_grouped(test["probs"], test_rec, trans, init)
    else:
        cal_probs, test_probs = cal["probs"], test["probs"]

    if args.score == "raps":
        k_reg, lam = tune_raps(cal_probs, cal["labels"], args.alpha)
        cal_scores = raps_score_matrix(cal_probs, k_reg=k_reg, lam=lam)
        test_scores = raps_score_matrix(test_probs, k_reg=k_reg, lam=lam)
        print(f"RAPS tuned to k_reg={k_reg} lam={lam}")
    else:
        k_reg, lam = None, None
        cal_scores = aps_score_matrix(cal_probs)
        test_scores = aps_score_matrix(test_probs)
    cal_true = cal_scores[np.arange(cal["labels"].size), cal["labels"]]

    global_q = split_conformal_quantile(cal_true, args.alpha)
    grid = np.asarray(args.lambda_grid, dtype=float)
    test_sl = subject_slices(test["subjects"])
    cal_subjects = np.unique(cal["subjects"])

    results = {
        "alpha": args.alpha,
        "smooth": bool(args.smooth),
        "score": args.score,
        "raps": None if k_reg is None else {"k_reg": k_reg, "lam": lam},
        "global_q": global_q,
        "n_cal_epochs": int(cal["labels"].size),
        "n_test_epochs": int(test["labels"].size),
        "cal_subjects": sorted(cal_subjects.tolist()),
        "test_subjects": sorted(test_sl),
        "budget_curve": {},
        "neighborhood": {},
    }

    # ---- warm-up budget curve -------------------------------------------
    for m in args.budget:
        lam, lam_detail = choose_lambda_loso(
            cal_scores,
            cal["labels"],
            cal["subjects"],
            global_q,
            m,
            args.alpha,
            grid,
            args.warmup_select,
        )
        sets_all, labels_all, subjects_all = [], [], []
        q_used, skipped = [], 0
        warm_rng = np.random.default_rng(777)
        for s, idx in test_sl.items():
            warm, eval_idx = split_warmup(idx, m, args.warmup_select, warm_rng)
            local_q = (
                split_conformal_quantile(
                    test_scores[warm, test["labels"][warm]], args.alpha
                )
                if m > 0
                else float("inf")
            )
            if m > 0 and not np.isfinite(local_q):
                skipped += 1
                q = global_q
            else:
                q = shrink_quantile(local_q, global_q, lam) if m > 0 else global_q
            if eval_idx.size == 0:
                continue
            sets_all.append(test_scores[eval_idx] <= q)
            labels_all.append(test["labels"][eval_idx])
            subjects_all.append(np.repeat(s, eval_idx.size))
            q_used.append(q)
        sets = np.concatenate(sets_all)
        labels = np.concatenate(labels_all)
        subjects = np.concatenate(subjects_all)
        results["budget_curve"][str(m)] = {
            "lambda": lam,
            "lambda_detail": lam_detail,
            "subjects_needing_global_fallback": skipped,
            "q_mean": float(np.mean(q_used)),
            **summarize(sets, labels, subjects, args.alpha),
        }

    # ---- neighbourhood (no labels from the test subject) ----------------
    cal_feat = np.stack(
        [subject_features(cal_probs[cal["subjects"] == s]) for s in cal_subjects]
    )
    for k in args.k:
        for lam in grid:
            lam = float(lam)
            sets_all, labels_all, subjects_all = [], [], []
            for s, idx in test_sl.items():
                feat = subject_features(test_probs[idx])
                dist = np.linalg.norm(cal_feat - feat[None, :], axis=1)
                neighbors = cal_subjects[np.argsort(dist)[: min(k, cal_subjects.size)]]
                pool = np.isin(cal["subjects"], neighbors)
                local_q = split_conformal_quantile(cal_true[pool], args.alpha)
                q = shrink_quantile(local_q, global_q, lam)
                sets_all.append(test_scores[idx] <= q)
                labels_all.append(test["labels"][idx])
                subjects_all.append(np.repeat(s, idx.size))
            results["neighborhood"][f"k{k}_shrink{lam:g}"] = summarize(
                np.concatenate(sets_all),
                np.concatenate(labels_all),
                np.concatenate(subjects_all),
                args.alpha,
            )

    # ---- oracle ----------------------------------------------------------
    sets_all, labels_all, subjects_all = [], [], []
    for s, idx in test_sl.items():
        q = split_conformal_quantile(test_scores[idx, test["labels"][idx]], args.alpha)
        sets_all.append(test_scores[idx] <= q)
        labels_all.append(test["labels"][idx])
        subjects_all.append(np.repeat(s, idx.size))
    results["oracle"] = summarize(
        np.concatenate(sets_all),
        np.concatenate(labels_all),
        np.concatenate(subjects_all),
        args.alpha,
    )

    # ---- adaptive conformal inference (ACI) ------------------------------
    # Long-run coverage on a stream of new subjects. alpha_t is updated after
    # every epoch; the threshold is the corresponding quantile of the pooled
    # calibration scores, read from a sorted array.
    if args.aci:
        rng = np.random.default_rng(0)
        sorted_cal = np.sort(cal_true)
        n_cal = sorted_cal.size
        a_t = args.alpha
        g0 = args.aci_gamma
        step = 0
        aci_sets, aci_labels, aci_subjects = [], [], []
        # Stream of new subjects in a random order.
        for s, idx in sorted(test_sl.items(), key=lambda kv: rng.random()):
            for t in idx:
                level = min(max(np.ceil((n_cal + 1) * (1.0 - a_t)) / n_cal, 0.0), 1.0)
                q = sorted_cal[min(int(level * (n_cal - 1)), n_cal - 1)]
                covered = bool(test_scores[t, test["labels"][t]] <= q)
                aci_sets.append(test_scores[t] <= q)
                aci_labels.append(test["labels"][t])
                aci_subjects.append(s)
                err = 0.0 if covered else 1.0
                step += 1
                a_t = float(np.clip(a_t + (g0 / np.sqrt(step)) * (args.alpha - err), 1e-3, 0.999))
        results["aci"] = summarize(
            np.asarray(aci_sets), np.asarray(aci_labels), np.asarray(aci_subjects), args.alpha
        )
        results["aci"]["final_alpha"] = a_t

    # ---- calibration-size curve -----------------------------------------
    # How many calibration subjects are needed before marginal coverage is
    # stable, and how much does per-subject coverage still vary? Repeated
    # random subsets of calibration subjects; the test subjects never change.
    cal_size_curve = {}
    rng = np.random.default_rng(0)
    for n_sub in args.cal_subjects:
        n_sub = min(int(n_sub), cal_subjects.size)
        covs, sizes, worsts = [], [], []
        for _ in range(args.cal_repeats):
            pick = rng.choice(cal_subjects, size=n_sub, replace=False)
            pool = np.isin(cal["subjects"], pick)
            q = split_conformal_quantile(cal_true[pool], args.alpha)
            sets = test_scores <= q
            covs.append(coverage(sets, test["labels"]))
            sizes.append(float(sets.sum(axis=1).mean()))
            worsts.append(
                min(
                    coverage(sets[idx], test["labels"][idx])
                    for idx in test_sl.values()
                )
            )
        cal_size_curve[str(n_sub)] = {
            "n_cal_subjects": n_sub,
            "repeats": args.cal_repeats,
            "coverage_mean": float(np.mean(covs)),
            "coverage_std": float(np.std(covs, ddof=1)) if len(covs) > 1 else 0.0,
            "set_size_mean": float(np.mean(sizes)),
            "worst_subject_mean": float(np.mean(worsts)),
            "worst_subject_p10": float(np.quantile(worsts, 0.1)),
        }
    results["cal_size_curve"] = cal_size_curve

    # ---- global, evaluated on the full test set --------------------------
    results["global"] = summarize(
        test_scores <= global_q, test["labels"], test["subjects"], args.alpha
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"alpha={args.alpha}  q_global={global_q:.4f}")
    print(f"{'method':<28}{'cov':>7}{'size':>7}{'sing':>7}{'worst':>8}{'p10':>7}")
    print("-" * 64)
    g = results["global"]
    print(
        f"{'global':<28}{g['coverage']:>7.3f}{g['set_size']:>7.2f}"
        f"{g['singleton_rate']:>7.3f}{g['worst_subject_coverage']:>8.3f}"
        f"{g['p10_subject_coverage']:>7.3f}"
    )
    o = results["oracle"]
    print(
        f"{'oracle_subject':<28}{o['coverage']:>7.3f}{o['set_size']:>7.2f}"
        f"{o['singleton_rate']:>7.3f}{o['worst_subject_coverage']:>8.3f}"
        f"{o['p10_subject_coverage']:>7.3f}"
    )
    for m, r in results["budget_curve"].items():
        print(
            f"{'sacp_m' + m: <28}{r['coverage']:>7.3f}{r['set_size']:>7.2f}"
            f"{r['singleton_rate']:>7.3f}{r['worst_subject_coverage']:>8.3f}"
            f"{r['p10_subject_coverage']:>7.3f}   lambda={r['lambda']:.2f}"
        )
    for name, r in results["neighborhood"].items():
        print(
            f"{'nbr_' + name: <28}{r['coverage']:>7.3f}{r['set_size']:>7.2f}"
            f"{r['singleton_rate']:>7.3f}{r['worst_subject_coverage']:>8.3f}"
            f"{r['p10_subject_coverage']:>7.3f}"
        )


if __name__ == "__main__":
    main()
