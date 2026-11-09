"""Aggregate per-fold/per-seed results into paper tables.

The convention is one run directory per (model, fold, seed), containing
``<model>_fold<f>_seed<s>.json`` plus ``cp_diagnostic.json``, ``sacp.json``
and/or ``temporal_cp.json``. Run directories can be nested arbitrarily under
the results root; the aggregator discovers them by filename.

Because the subject-wise folds partition the cohort (each subject is tested
exactly once), per-subject coverage is merged across folds before computing
worst-subject and p10 statistics. That is the number the paper should report.

Usage:
    python scripts/aggregate.py --root results --out results/summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover(
    root: Path, filename: str, exclude: Path | list[Path] | None = None
) -> list[Path]:
    """Files named ``filename`` under ``root``, skipping summary directories.

    A summary directory typically lives inside the results root, so without
    the exclusion a second aggregation pass reads its own output back in and
    treats formatted strings such as ``"0.899 \\pm 0.010"`` as numbers. Every
    aggregate in this module excludes ``<root>/summary`` as well as the
    requested output directory, so writing the summary elsewhere cannot
    silently reintroduce the problem.
    """
    found = sorted(root.rglob(filename))
    if exclude is None:
        return found
    excluded = [exclude] if isinstance(exclude, Path) else list(exclude)
    roots = {p.resolve() for p in excluded}
    kept = []
    for path in found:
        resolved = path.resolve()
        if any(resolved == r or r in resolved.parents for r in roots):
            continue
        kept.append(path)
    return kept


def fmt_ci(values: list[float]) -> str:
    if not values:
        return "-"
    a = np.asarray(values, dtype=float)
    return f"{a.mean():.3f} ± {a.std(ddof=1) if a.size > 1 else 0.0:.3f}"


def merge_subject_coverage(entries: list[dict]) -> dict[str, float]:
    """One coverage value per subject, averaged over every run that has it.

    Each subject is tested once per (fold, seed, model) combination --- the
    folds partition the cohort and the seeds and backbones re-run the same
    partition --- so a plain dict update would silently keep whichever run was
    read last. Averaging is what makes "each subject appears exactly once"
    true in the reported worst-patient and dispersion statistics.
    """
    total: dict[str, float] = {}
    count: dict[str, int] = {}
    for entry in entries:
        for subject, value in entry.get("per_subject_coverage", {}).items():
            total[subject] = total.get(subject, 0.0) + float(value)
            count[subject] = count.get(subject, 0) + 1
    return {s: total[s] / count[s] for s in total}


def coverage_stats(per_subject: dict[str, float], alpha: float) -> dict:
    if not per_subject:
        return {}
    values = np.asarray(list(per_subject.values()), dtype=float)
    return {
        "n_subjects": int(values.size),
        "mean_subject_coverage": float(values.mean()),
        "worst_subject_coverage": float(values.min()),
        "p10_subject_coverage": float(np.quantile(values, 0.1)),
        "std_subject_coverage": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        # Dispersion around the nominal level. The violation rate is brittle
        # when a method deliberately pulls every patient to exactly 1 - alpha:
        # half of them end up a hair below. This statistic does not care which
        # side of nominal a patient falls on, only how far away it is, and is
        # the one to read when comparing equalising methods.
        "mean_abs_dev_from_nominal": float(np.abs(values - (1.0 - alpha)).mean()),
        "range_subject_coverage": float(values.max() - values.min()),
        "subject_violation_rate": float((values < 1.0 - alpha).mean()),
    }


def aggregate_training(root: Path, exclude: Path | None = None) -> list[dict]:
    rows = []
    for path in discover(root, "*.json", exclude):
        name = path.name
        if "_fold" not in name or "_seed" not in name:
            continue
        data = load(path)
        if "test_metrics" not in data:
            continue
        rows.append(
            {
                "model": data.get("model", "?"),
                "fold": data["fold"],
                "seed": data["seed"],
                "accuracy": data["test_metrics"]["accuracy"],
                "kappa": data["test_metrics"]["kappa"],
                "mf1": data["test_metrics"]["mf1"],
                "ece": data["test_metrics"]["ece"],
                "n_test": data.get("n_test", 0),
            }
        )
    return rows


def sacp_variant(path: Path) -> str:
    """Variant label for a sacp output file (raw, smoothed, or RAPS)."""
    stem = path.stem
    if stem == "sacp":
        return "aps"
    return stem[len("sacp_"):] if stem.startswith("sacp_") else stem


def aggregate_sacp(root: Path, alpha: float, exclude: Path | None = None) -> dict:
    files = discover(root, "sacp*.json", exclude)
    by_method: dict[str, list[dict]] = defaultdict(list)
    # Shrinkage weights chosen by LOSO, per score family and budget. This is a
    # property of the procedure rather than a performance number, so it is
    # recorded separately: it is the evidence that the selection criterion is
    # degenerate at this cohort size.
    selection: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        data = load(path)
        variant = sacp_variant(path)
        prefix = f"{variant}/"
        global_entry = data.get("global")
        if global_entry:
            by_method[f"{prefix}global"].append(global_entry)
        for m, entry in data.get("budget_curve", {}).items():
            by_method[f"{prefix}sacp_m{m}"].append(entry)
            if entry.get("lambda") is not None:
                selection[variant][m].append(float(entry["lambda"]))
        for name, entry in data.get("neighborhood", {}).items():
            by_method[f"{prefix}nbr_{name}"].append(entry)
        if data.get("oracle"):
            by_method[f"{prefix}oracle_subject"].append(data["oracle"])

    summary = {}
    for method, entries in by_method.items():
        per_subject = merge_subject_coverage(entries)
        stats = coverage_stats(per_subject, alpha)
        sets = [e.get("set_size", np.nan) for e in entries]
        sing = [e.get("singleton_rate", np.nan) for e in entries]
        work = [e.get("review_workload", np.nan) for e in entries]
        cov = [e.get("coverage", np.nan) for e in entries]
        summary[method] = {
            "n_runs": len(entries),
            **stats,
            "marginal_coverage": fmt_ci([c for c in cov if np.isfinite(c)]),
            "set_size": fmt_ci([s for s in sets if np.isfinite(s)]),
            "singleton_rate": fmt_ci([s for s in sing if np.isfinite(s)]),
            "review_workload": fmt_ci([w for w in work if np.isfinite(w)]),
        }
    sel_out = {
        variant: {
            m: {
                "n": len(vals),
                "frac_zero": float(np.mean(np.asarray(vals) == 0.0)),
                "frac_one": float(np.mean(np.asarray(vals) == 1.0)),
            }
            for m, vals in sorted(by_m.items(), key=lambda kv: int(kv[0]))
        }
        for variant, by_m in selection.items()
    }
    # Joint degeneracy: how many run-variant pairs pick lambda=0 at *every*
    # budget up to m=80. The per-budget rates above cannot recover this, because
    # the same run may select zero at one budget and non-zero at the next, and
    # the paper quotes the joint count as its evidence that the weight is not
    # identified at four calibration subjects.
    joint = {}
    for variant, by_m in selection.items():
        n_runs = max((len(v) for v in by_m.values()), default=0)
        all_zero = [True] * n_runs
        for m, vals in by_m.items():
            if int(m) <= 80:
                for i, lam in enumerate(vals[:n_runs]):
                    if lam != 0.0:
                        all_zero[i] = False
        joint[variant] = {
            "n_runs": n_runs,
            "n_zero_through_80": int(sum(all_zero)),
        }
    return {
        "n_sacp_files": len(files),
        "variants": sorted({sacp_variant(p) for p in files}),
        "selection": sel_out,
        "selection_joint": joint,
        "methods": summary,
    }


def aggregate_temporal(root: Path, alpha: float, exclude: Path | None = None) -> dict:
    files = discover(root, "temporal_cp.json", exclude)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        data = load(path)
        for name, entry in data.get("methods", {}).items():
            if "error" in entry:
                continue
            by_method[name].append(entry)

    summary = {}
    for method, entries in by_method.items():
        per_subject = merge_subject_coverage(entries)
        stats = coverage_stats(per_subject, alpha)
        def ci(key):
            vals = [e.get(key, np.nan) for e in entries]
            return fmt_ci([v for v in vals if np.isfinite(v)])
        summary[method] = {
            "n_runs": len(entries),
            **stats,
            "marginal_coverage": ci("coverage"),
            "set_size": ci("set_size"),
            "singleton_rate": ci("singleton_rate"),
            "empty_rate": ci("empty_rate"),
            "review_workload": ci("review_workload"),
            "temporal_consistency": ci("temporal_consistency"),
            "sequence_coverage": ci("sequence_coverage"),
        }
        # Per-class coverage is a list ordered by class index
        # (W, N1, N2, N3, REM) in each entry; average it class-wise so the
        # score's class-level trade-off can be tabulated.
        per_class: dict[int, list[float]] = defaultdict(list)
        for entry in entries:
            raw = entry.get("per_class_coverage", [])
            items = enumerate(raw) if isinstance(raw, list) else raw.items()
            for cls, value in items:
                if value is not None and np.isfinite(value):
                    per_class[int(cls)].append(float(value))
        summary[method]["per_class_coverage"] = {
            int(cls): float(np.mean(vals)) for cls, vals in sorted(per_class.items())
        }
    return {"n_temporal_files": len(files), "methods": summary}


def aggregate_aci(root: Path, alpha: float, exclude: Path | None = None) -> dict:
    """Per-subject ACI results, plus the LOSO-chosen step size per score."""
    files = discover(root, "subject_aci.json", exclude)
    by_method: dict[str, list[dict]] = defaultdict(list)
    selected: dict[str, list[float]] = defaultdict(list)
    for path in files:
        data = load(path)
        for name, entry in data.get("methods", {}).items():
            if "error" in entry:
                continue
            by_method[name].append(entry)
        for key, value in data.items():
            if key.startswith("selected_gamma0_"):
                selected[key[len("selected_gamma0_"):]].append(float(value))

    summary = {}
    for method, entries in by_method.items():
        # Subject-wise folds partition the cohort, so per-subject coverage can
        # be merged across runs (each subject is tested exactly once per seed).
        per_subject = merge_subject_coverage(entries)
        stats = coverage_stats(per_subject, alpha)

        def ci(key):
            vals = [e.get(key, np.nan) for e in entries]
            return fmt_ci([v for v in vals if np.isfinite(v)])

        summary[method] = {
            "n_runs": len(entries),
            **stats,
            "marginal_coverage": ci("coverage"),
            "set_size": ci("set_size"),
            "singleton_rate": ci("singleton_rate"),
            "review_workload": ci("review_workload"),
            "temporal_consistency": ci("temporal_consistency"),
        }
    return {
        "n_files": len(files),
        "selected_gamma0": {k: {"mean": float(np.mean(v)), "values": v} for k, v in selected.items()},
        "methods": summary,
    }


def aggregate_diagnostic(root: Path, exclude: Path | None = None, alpha: float = 0.1) -> dict:
    """Pooled, Mondrian, neighbourhood, oracle and random-epoch baselines.

    Subject-wise folds partition the cohort, so per-subject coverage is merged
    across folds, exactly as for the main and ACI tables; a fold-averaged worst
    patient would otherwise be biased upward by the noise of a four-subject
    worst-case. The random-epoch row has no per-subject decomposition --- the
    split mixes epochs from every subject --- so it is reported as a mean over
    repeats and runs.
    """
    files = discover(root, "cp_diagnostic.json", exclude)
    by_method: dict[str, list[dict]] = defaultdict(list)
    random_rows: list[dict] = []
    for path in files:
        data = load(path)
        for name, entry in data.get("methods", {}).items():
            if name == "random_epoch":
                continue
            by_method[name].append(entry)
        if "random_epoch" in data.get("methods", {}):
            random_rows.append(data["methods"]["random_epoch"])

    def spread(values: list[float]) -> dict:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "sd": float("nan")}
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}

    out = {}
    for name, entries in by_method.items():
        per_subject = merge_subject_coverage(entries)
        stats = coverage_stats(per_subject, alpha)
        out[name] = {
            **stats,
            "n_runs": len(entries),
            "coverage": spread([e.get("coverage", np.nan) for e in entries]),
            "set_size": spread([e.get("set_size", np.nan) for e in entries]),
            "singleton_rate": spread([e.get("singleton_rate", np.nan) for e in entries]),
            "empty_rate": spread([e.get("empty_rate", np.nan) for e in entries]),
            "review_workload": spread([e.get("review_workload", np.nan) for e in entries]),
        }
        # Per-class coverage is a dict {class: value} in each entry; average
        # class-wise across runs so the N1 weakness can be quoted in text.
        # ``per_class_coverage`` is stored as a list ordered by class index
        # (W, N1, N2, N3, REM); older runs used a dict.
        per_class: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            raw = entry.get("per_class_coverage", {})
            items = enumerate(raw) if isinstance(raw, list) else raw.items()
            for cls, value in items:
                if value is not None and np.isfinite(value):
                    per_class[str(cls)].append(float(value))
        out[name]["per_class_coverage"] = {
            cls: spread(vals)["mean"] for cls, vals in per_class.items()
        }
    if random_rows:
        out["random_epoch"] = {
            "coverage": spread([r.get("coverage_mean", np.nan) for r in random_rows]),
            "set_size": spread([r.get("set_size_mean", np.nan) for r in random_rows]),
            "coverage_repeat_sd": spread([r.get("coverage_std", np.nan) for r in random_rows]),
            "n_runs": len(random_rows),
        }
    return {"n_files": len(files), "methods": out}


def aggregate_cohort_size(root: Path, exclude: Path | None = None) -> dict:
    """Coverage curves against the number of calibration subjects.

    Every run writes one ``cohort_size.json``; the curves are averaged over
    runs, and the spread reported is the between-run spread of the subsampled
    means (the within-run spread over subsamples is an order of magnitude
    smaller and is kept under ``"subsample_sd"`` for reference).
    """
    files = discover(root, "cohort_size.json", exclude)
    curves: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    sizes: list[int] = []
    for path in files:
        data = load(path)
        if data.get("n_cal_subjects"):
            sizes.append(int(data["n_cal_subjects"]))
        for score, curve in data.get("curves", {}).items():
            for k, stats in curve.items():
                curves[score][int(k)].append(stats)

    summary = {}
    for score, by_k in curves.items():
        summary[score] = {}
        for k in sorted(by_k):
            entries = by_k[k]
            summary[score][str(k)] = {
                key: {
                    "mean": float(np.mean([e[key]["mean"] for e in entries])),
                    "sd": float(np.std([e[key]["mean"] for e in entries], ddof=1))
                    if len(entries) > 1
                    else 0.0,
                    "subsample_sd": float(np.mean([e[key]["sd"] for e in entries])),
                }
                for key in ("coverage", "worst", "p10")
            }
    return {
        "n_files": len(files),
        "n_cal_subjects": sorted(set(sizes)),
        "curves": summary,
    }


def write_markdown(path: Path, title: str, table: dict[str, dict], columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    header = "| method | " + " | ".join(columns) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(columns) + 1))
    for method, stats in table.items():
        cells = []
        for c in columns:
            value = stats.get(c, "-")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(str(value))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/summary"))
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Never read a previous aggregation back in: exclude both the requested
    # output directory and the conventional <root>/summary location.
    exclude: list[Path] = [args.out]
    default_summary = args.root / "summary"
    if default_summary.resolve() != args.out.resolve():
        exclude.append(default_summary)

    training = aggregate_training(args.root, exclude)
    if training:
        by_model: dict[str, list[dict]] = defaultdict(list)
        for row in training:
            by_model[row["model"]].append(row)
        lines = ["# Classification performance", "", "| model | acc | kappa | MF1 | ECE | runs |", "|---|---|---|---|---|---|"]
        for model, rows in sorted(by_model.items()):
            def ci(key):
                a = np.asarray([r[key] for r in rows], dtype=float)
                return f"{a.mean():.3f} ± {a.std(ddof=1) if a.size > 1 else 0.0:.3f}"
            lines.append(
                f"| {model} | {ci('accuracy')} | {ci('kappa')} | {ci('mf1')} | {ci('ece')} | {len(rows)} |"
            )
        (args.out / "training.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with open(args.out / "training.json", "w", encoding="utf-8") as f:
            json.dump(training, f, indent=2)

    sacp = aggregate_sacp(args.root, args.alpha, exclude)
    if sacp["methods"]:
        write_markdown(
            args.out / "sacp.md",
            "Subject-adaptive conformal prediction",
            sacp["methods"],
            [
                "n_runs",
                "marginal_coverage",
                "worst_subject_coverage",
                "p10_subject_coverage",
                "subject_violation_rate",
                "set_size",
                "singleton_rate",
                "review_workload",
            ],
        )
        with open(args.out / "sacp.json", "w", encoding="utf-8") as f:
            json.dump(sacp, f, indent=2)

    temporal = aggregate_temporal(args.root, args.alpha, exclude)
    if temporal["methods"]:
        write_markdown(
            args.out / "temporal.md",
            "Temporal conformal prediction",
            temporal["methods"],
            [
                "n_runs",
                "marginal_coverage",
                "worst_subject_coverage",
                "p10_subject_coverage",
                "set_size",
                "review_workload",
                "temporal_consistency",
                "sequence_coverage",
            ],
        )
        with open(args.out / "temporal.json", "w", encoding="utf-8") as f:
            json.dump(temporal, f, indent=2)

    aci = aggregate_aci(args.root, args.alpha, exclude)
    if aci["methods"]:
        write_markdown(
            args.out / "subject_aci.md",
            "Per-subject adaptive conformal inference",
            aci["methods"],
            [
                "n_runs",
                "marginal_coverage",
                "worst_subject_coverage",
                "p10_subject_coverage",
                "subject_violation_rate",
                "set_size",
                "singleton_rate",
                "temporal_consistency",
            ],
        )
        with open(args.out / "subject_aci.json", "w", encoding="utf-8") as f:
            json.dump(aci, f, indent=2)

    diagnostic = aggregate_diagnostic(args.root, exclude, args.alpha)
    if diagnostic["methods"]:
        with open(args.out / "diagnostic.json", "w", encoding="utf-8") as f:
            json.dump(diagnostic, f, indent=2)

    cohort = aggregate_cohort_size(args.root, exclude)
    if cohort["curves"]:
        with open(args.out / "cohort_size.json", "w", encoding="utf-8") as f:
            json.dump(cohort, f, indent=2)

    print(f"wrote summaries to {args.out}")


if __name__ == "__main__":
    main()
