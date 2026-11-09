"""Decompose the per-patient coverage deficit into prior and score components.

For a patient whose epochs have class prior :math:`\\pi_{s,k}` and whose
class-conditional coverage under the pooled split-conformal threshold is
:math:`r_{s,k}`, the patient's coverage is exactly

    cov_s = sum_k pi_{s,k} r_{s,k}.

Writing :math:`\\bar\\pi_k` and :math:`\\bar r_k` for the pooled test-set
versions, the deviation of a patient's coverage from the pooled coverage
splits into three terms with no remainder::

    cov_s - cov_pool = sum_k (pi_sk - pi_k) r_k         (prior term)
                     + sum_k pi_k (r_sk - r_k)         (score term)
                     + sum_k (pi_sk - pi_k)(r_sk - r_k) (interaction)

The prior term is what a difference in the label mix alone would do; the score
term is what a difference in how the network ranks the true stage would do at
a fixed label mix. The script also records two label-free quantities per
patient --- the mean prediction-set size and the mean confidence --- and asks
how much of the coverage variation they explain; if the score term dominates,
the deficit is a confidence-scale effect that a clinician's audit could see
without any labels.

Example
-------
python scripts/calibration_decomposition.py --root results/main \
    --out results/main/summary/decomposition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sleepcp.conformal import (  # noqa: E402
    aps_score_matrix,
    aps_scores,
    split_conformal_quantile,
)

NUM_CLASSES = 5


def load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(probs, labels, subjects)`` for the test half of one run."""
    test = sorted(run_dir.glob("*_test.npz"))
    if not test:
        return None
    with np.load(test[0], allow_pickle=False) as data:
        return (
            data["probs"].astype(np.float64),
            data["labels"].astype(np.int64),
            data["subjects"].astype(str),
        )


def load_calibration(run_dir: Path) -> np.ndarray | None:
    cal = sorted(run_dir.glob("*_cal.npz"))
    if not cal:
        return None
    with np.load(cal[0], allow_pickle=False) as data:
        probs = data["probs"].astype(np.float64)
        labels = data["labels"].astype(np.int64)
    return aps_scores(probs, labels)


def class_rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Class-conditional coverage; classes with no epochs get ``nan``."""
    rates = np.full(NUM_CLASSES, np.nan)
    covered = scores <= threshold
    for k in range(NUM_CLASSES):
        mask = labels == k
        if mask.any():
            rates[k] = covered[mask].mean()
    return rates


def class_prior(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    return counts / counts.sum()


def weighted(prior: np.ndarray, rates: np.ndarray) -> float:
    """Class-prior weighted coverage, ignoring classes the subject lacks."""
    mask = ~np.isnan(rates)
    if not mask.any():
        return float("nan")
    p = prior[mask]
    return float((p * rates[mask]).sum() / p.sum())


def analyse_run(run_dir: Path, alpha: float) -> dict | None:
    loaded = load_run(run_dir)
    if loaded is None:
        return None
    probs, labels, subjects = loaded
    cal_scores = load_calibration(run_dir)
    if cal_scores is None:
        return None
    threshold = split_conformal_quantile(cal_scores, alpha)

    scores = aps_scores(probs, labels)
    score_matrix = aps_score_matrix(probs)
    set_sizes = (score_matrix <= threshold).sum(axis=1)

    pooled_prior = class_prior(labels)
    pooled_rates = class_rates(scores, labels, threshold)
    pooled_coverage = weighted(pooled_prior, pooled_rates)

    rows = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        prior_s = class_prior(labels[mask])
        rates_s = class_rates(scores[mask], labels[mask], threshold)
        valid = ~np.isnan(rates_s) & ~np.isnan(pooled_rates)
        if not valid.any():
            continue
        # Restrict the decomposition to classes this patient actually has,
        # renormalising both priors on that support so the identity holds.
        p_s = prior_s[valid] / prior_s[valid].sum()
        p_ref = pooled_prior[valid] / pooled_prior[valid].sum()
        r_s = rates_s[valid]
        r_ref = pooled_rates[valid]
        coverage = float((p_s * r_s).sum())
        prior_term = float(((p_s - p_ref) * r_ref).sum())
        score_term = float((p_ref * (r_s - r_ref)).sum())
        interaction = float(((p_s - p_ref) * (r_s - r_ref)).sum())
        # Reference coverage on the same label support, so the identity holds
        # exactly even when the patient is missing a stage.
        pooled_support = float((p_ref * r_ref).sum())

        # Location and spread of the within-class score distribution relative
        # to the pooled one, on the classes the patient has.
        median_shift, iqr_shift = [], []
        for k in np.flatnonzero(valid):
            subject_scores = scores[mask & (labels == k)]
            pooled_scores = scores[labels == k]
            if subject_scores.size < 10:
                continue
            median_shift.append(
                float(np.median(subject_scores) - np.median(pooled_scores))
            )
            iqr_shift.append(
                float(
                    np.subtract(*np.quantile(subject_scores, [0.75, 0.25]))
                    - np.subtract(*np.quantile(pooled_scores, [0.75, 0.25]))
                )
            )

        rows.append(
            dict(
                subject=str(subject),
                n=int(mask.sum()),
                coverage=coverage,
                prior_term=prior_term,
                score_term=score_term,
                interaction=interaction,
                residual=coverage - pooled_support
                - prior_term - score_term - interaction,
                mean_set_size=float(set_sizes[mask].mean()),
                mean_confidence=float(probs[mask].max(axis=1).mean()),
                mean_score=float(scores[mask].mean()),
                accuracy=float((probs[mask].argmax(axis=1) == labels[mask]).mean()),
                prior_l1=float(np.abs(prior_s - pooled_prior).sum() / 2.0),
                median_shift=float(np.mean(median_shift)) if median_shift else float("nan"),
                iqr_shift=float(np.mean(iqr_shift)) if iqr_shift else float("nan"),
            )
        )

    coverage = np.array([r["coverage"] for r in rows])
    terms = {
        "prior": np.array([r["prior_term"] for r in rows]),
        "score": np.array([r["score_term"] for r in rows]),
        "interaction": np.array([r["interaction"] for r in rows]),
    }
    set_size = np.array([r["mean_set_size"] for r in rows])
    accuracy = np.array([r["accuracy"] for r in rows])
    prior_l1 = np.array([r["prior_l1"] for r in rows])

    def correlate(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return dict(
        run=str(run_dir),
        threshold=float(threshold),
        pooled_coverage=float(pooled_coverage),
        n_subjects=len(rows),
        coverage_sd=float(coverage.std(ddof=1)) if len(rows) > 1 else 0.0,
        worst_coverage=float(coverage.min()),
        mean_abs_terms={k: float(np.mean(np.abs(v))) for k, v in terms.items()},
        score_dominant_fraction=float(np.mean(np.abs(terms["score"]) > np.abs(terms["prior"]))),
        max_abs_residual=float(np.max(np.abs([r["residual"] for r in rows]))),
        correlations=dict(
            coverage_vs_set_size=correlate(coverage, set_size),
            coverage_vs_accuracy=correlate(coverage, accuracy),
            coverage_vs_prior_l1=correlate(coverage, prior_l1),
        ),
        location_vs_spread=dict(
            mean_abs_median_shift=float(np.nanmean(np.abs([r["median_shift"] for r in rows]))),
            mean_abs_iqr_shift=float(np.nanmean(np.abs([r["iqr_shift"] for r in rows]))),
        ),
        rows=rows,
    )


def merge_runs(runs: list[dict]) -> list[dict]:
    """Merge the folds of one (model, seed) into a per-patient table.

    Each subject is tested in exactly one fold, so the folds partition the
    cohort; merging them gives one row per patient and makes the correlations
    comparable to the paper's per-patient statistics.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        path = Path(run["run"])
        model = path.parent.name
        seed = path.name.rsplit("_seed", 1)[-1]
        groups.setdefault((model, seed), []).extend(run["rows"])

    # The paper's per-patient statistics average each patient over every run
    # that tests them (all models and seeds), then take the spread. The same
    # averaging is applied here so the decomposition is read on the paper's
    # scale.
    by_subject: dict[str, list[dict]] = {}
    for rows in groups.values():
        for row in rows:
            by_subject.setdefault(row["subject"], []).append(row)
    averaged = []
    for subject, entries in sorted(by_subject.items()):
        mean = {
            key: float(np.mean([e[key] for e in entries]))
            for key in ("coverage", "prior_term", "score_term", "interaction",
                        "mean_set_size", "accuracy", "prior_l1")
        }
        averaged.append(dict(subject=subject, **mean))
    groups[("all", "all")] = averaged

    merged = []
    for (model, seed), rows in sorted(groups.items()):
        coverage = np.array([r["coverage"] for r in rows])
        terms = {
            "prior": np.array([r["prior_term"] for r in rows]),
            "score": np.array([r["score_term"] for r in rows]),
            "interaction": np.array([r["interaction"] for r in rows]),
        }
        set_size = np.array([r["mean_set_size"] for r in rows])
        accuracy = np.array([r["accuracy"] for r in rows])
        prior_l1 = np.array([r["prior_l1"] for r in rows])

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        merged.append(
            dict(
                model=model,
                seed=seed,
                n_subjects=len(rows),
                coverage_sd=float(coverage.std(ddof=1)),
                worst_coverage=float(coverage.min()),
                mean_abs_terms={k: float(np.mean(np.abs(v))) for k, v in terms.items()},
                score_dominant_fraction=float(
                    np.mean(np.abs(terms["score"]) > np.abs(terms["prior"]))
                ),
                correlations=dict(
                    coverage_vs_set_size=corr(coverage, set_size),
                    coverage_vs_accuracy=corr(coverage, accuracy),
                    coverage_vs_prior_l1=corr(coverage, prior_l1),
                ),
            )
        )
    return merged


def summarise(runs: list[dict], merged: list[dict]) -> dict:
    return dict(
        merged=dict(
            n_groups=len(merged),
            n_subjects=float(np.mean([m["n_subjects"] for m in merged])),
            coverage_sd=float(np.mean([m["coverage_sd"] for m in merged])),
            worst_coverage=float(np.mean([m["worst_coverage"] for m in merged])),
            mean_abs_terms={
                k: float(np.mean([m["mean_abs_terms"][k] for m in merged]))
                for k in ("prior", "score", "interaction")
            },
            score_dominant_fraction=float(
                np.mean([m["score_dominant_fraction"] for m in merged])
            ),
            correlations={
                k: float(np.nanmean([m["correlations"][k] for m in merged]))
                for k in ("coverage_vs_set_size", "coverage_vs_accuracy",
                          "coverage_vs_prior_l1")
            },
        ),
        n_runs=len(runs),
        coverage_sd=float(np.mean([r["coverage_sd"] for r in runs])),
        worst_coverage=float(np.mean([r["worst_coverage"] for r in runs])),
        mean_abs_terms={
            k: float(np.mean([r["mean_abs_terms"][k] for r in runs]))
            for k in ("prior", "score", "interaction")
        },
        score_dominant_fraction=float(
            np.mean([r["score_dominant_fraction"] for r in runs])
        ),
        correlations={
            k: float(np.nanmean([r["correlations"][k] for r in runs]))
            for k in ("coverage_vs_set_size", "coverage_vs_accuracy", "coverage_vs_prior_l1")
        },
        location_vs_spread={
            k: float(np.mean([r["location_vs_spread"][k] for r in runs]))
            for k in ("mean_abs_median_shift", "mean_abs_iqr_shift")
        },
        max_abs_residual=float(max(r["max_abs_residual"] for r in runs)),
        n_subjects=float(np.mean([r["n_subjects"] for r in runs])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(p for p in args.root.rglob("*_seed*") if p.is_dir()):
        result = analyse_run(run_dir, args.alpha)
        if result is None:
            continue
        runs.append(result)
        terms = result["mean_abs_terms"]
        corr = result["correlations"]
        print(
            f"{run_dir.relative_to(args.root)}: n={result['n_subjects']} "
            f"worst={result['worst_coverage']:.3f} |prior|={terms['prior']:.3f} "
            f"|score|={terms['score']:.3f} |int|={terms['interaction']:.3f} "
            f"r(set)={corr['coverage_vs_set_size']:+.2f} "
            f"r(acc)={corr['coverage_vs_accuracy']:+.2f}",
            flush=True,
        )
    if not runs:
        raise SystemExit(f"no runs under {args.root}")
    merged = merge_runs(runs)
    summary = summarise(runs, merged)
    payload = dict(summary=summary, merged=merged, runs=runs)
    out = args.out or args.root / "summary" / "decomposition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
