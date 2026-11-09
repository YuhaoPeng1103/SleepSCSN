"""Render the LaTeX tables of the normalisation paper from the result JSONs.

Reads ``results/norm/arms.json`` (and, when present, the HMC and cross-cohort
summaries) and writes ``paper2/tables/*.tex`` so that the manuscript never has
a hand-typed number.

Example
-------
python scripts/make_tables2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = [("batch", "Batch (cohort)"), ("group", "Group (per sample)"),
        ("adabn", "Batch + AdaBN (per recording)")]
MODELS = [("eegnet", "EEGNet"), ("sleepcnn", "SleepCNN")]

COLUMNS = [
    ("accuracy", "Acc."), ("kappa", "$\\kappa$"), ("mf1", "mF1"),
    ("ece", "ECE"), ("coverage", "Cov."), ("set_size", "Set"),
    ("workload", "Wkld"), ("worst", "Worst"), ("coverage_sd", "SD"),
    ("probe", "Probe"),
]


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def main_table(summary: dict, models=MODELS, first_column: str = "Backbone") -> str:
    header = " & ".join([first_column, "Normalisation"]
                        + [f"\\textbf{{{name}}}" for _, name in COLUMNS]) + " \\\\"
    lines = [
        "\\begin{tabular}{llrrrrrrrrrr}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    present = [(mk, mn) for mk, mn in models
               if any(summary.get(a, {}).get(mk) for a, _ in ARMS)]
    for pos, (model_key, model_name) in enumerate(present):
        arms = [(a, n) for a, n in ARMS if summary.get(a, {}).get(model_key)]
        for i, (arm_key, arm_name) in enumerate(arms):
            stats = summary[arm_key][model_key]
            cells = []
            for metric, _ in COLUMNS:
                digits = 2 if metric in {"set_size"} else 3
                cells.append(fmt(stats.get(metric), digits))
            label = (f"\\multirow{{{len(arms)}}}{{*}}{{{model_name}}}"
                     if i == 0 else "")
            lines.append(f"{label} & {arm_name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\bottomrule" if pos == len(present) - 1 else "\\midrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def cross_cohort_table(paths: list[tuple[Path, str]]) -> str:
    """Accuracy of a source-cohort model on the target cohort, two modes."""
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Backbone & Statistics & Acc. & $\\kappa$ & mF1 & ECE & Worst rec. \\\\",
        "\\midrule",
    ]
    rows = []
    for path, label in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for mode, mode_name in (("cohort", "Source (cohort)"),
                                ("recording", "Target (AdaBN)")):
            s = data.get(mode)
            if not s:
                continue
            rows.append(f"{label} & {mode_name} & {s['accuracy']:.3f} & "
                        f"{s['kappa']:.3f} & {s['macro_f1']:.3f} & "
                        f"{s['ece']:.3f} & {s['recording_accuracy_worst']:.3f} \\\\")
    if not rows:
        return "% missing: cross-cohort results"
    lines += rows + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


PAIR_ROWS = [
    ("accuracy", "Accuracy", 3), ("set_size", "Set size", 3),
    ("workload", "Review workload", 3), ("worst", "Worst-patient coverage", 3),
    ("coverage_sd", "Coverage SD", 4), ("probe", "Subject probe", 3),
]


def pair_table(pairs: dict, name: str) -> str:
    stats = pairs.get(name)
    if not stats:
        return "% missing: " + name
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Quantity & Batch & Other & Difference [95\\% CI] & $d_z$ & $p$ \\\\",
        "\\midrule",
    ]
    for metric, label, digits in PAIR_ROWS:
        s = stats.get(metric)
        if not s:
            continue
        lo, hi = s["ci95"]
        diff = f"{s['diff']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"
        lines.append(f"{label} & {fmt(s['a_mean'], digits)} & {fmt(s['b_mean'], digits)} "
                      f"& {diff} & {s['dz']:+.2f} & {s['wilcoxon_p']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def rows_block(pairs: dict, name: str) -> list[str]:
    """Body rows of one paired comparison."""
    stats = pairs.get(name)
    if not stats:
        return [f"% missing: {name}"]
    lines = []
    for metric, label, digits in PAIR_ROWS:
        s = stats.get(metric)
        if not s:
            continue
        lo, hi = s["ci95"]
        diff = f"{s['diff']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"
        lines.append(f"{label} & {fmt(s['a_mean'], digits)} & {fmt(s['b_mean'], digits)} "
                     f"& {diff} & {s['dz']:+.2f} & {s['wilcoxon_p']:.4f} \\\\")
    return lines


def combined_table(pairs: dict, entries: list[tuple[str, str]],
                   first_header: str, other_header: str) -> str:
    """One tabular holding several paired comparisons, separated by backbones.

    ``entries`` are ``(pair key, block title)``. Everything is written in a
    single file because a ``\\input`` inside a ``tabular`` does not work.
    """
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        f"Quantity & {first_header} & {other_header} & Difference [95\\% CI] & $d_z$ & $p$ \\\\",
        "\\midrule",
    ]
    for i, (key, title) in enumerate(entries):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}\\\\")
        lines += rows_block(pairs, key)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=Path, default=Path("results/norm/arms.json"))
    parser.add_argument("--hmc", type=Path, default=Path("results/norm/arms_hmc.json"))
    parser.add_argument("--hmc-sleepcnn", type=Path,
                        default=Path("results/norm/arms_hmc_sleepcnn.json"))
    parser.add_argument("--cross", type=Path,
                        default=Path("results/norm/cross_cohort"))
    parser.add_argument("--out", type=Path, default=Path("paper2/tex/tables"))
    args = parser.parse_args()

    data = json.loads(args.arms.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sleep_edf_arms.tex").write_text(main_table(data["summary"]), encoding="utf-8")

    if args.hmc.exists():
        hmc = json.loads(args.hmc.read_text(encoding="utf-8"))
        summary, pairs = dict(hmc["summary"]), dict(hmc["pairs"])
        # The larger backbone is aggregated into its own file because its
        # runs are paired within the backbone only.
        if args.hmc_sleepcnn.exists():
            extra = json.loads(args.hmc_sleepcnn.read_text(encoding="utf-8"))
            for arm, by_model in extra["summary"].items():
                summary.setdefault(arm, {}).update(by_model)
            pairs.update(extra["pairs"])
        models = [("eegnet", "EEGNet")]
        if any(summary.get(a, {}).get("sleepcnn") for a in ("batch", "adabn")):
            models.append(("sleepcnn", "SleepCNN"))
        (args.out / "hmc_arms.tex").write_text(
            main_table(summary, models=models, first_column="Backbone"),
            encoding="utf-8")
        entries = [("eegnet|batch-vs-group", "Batch vs group"),
                   ("eegnet|batch-vs-adabn", "Batch vs AdaBN")]
        if "sleepcnn|batch-vs-adabn" in pairs:
            entries = [("eegnet|batch-vs-group", "EEGNet: batch vs group"),
                       ("eegnet|batch-vs-adabn", "EEGNet: batch vs AdaBN"),
                       ("sleepcnn|batch-vs-adabn", "SleepCNN: batch vs AdaBN")]
        (args.out / "hmc_pairs.tex").write_text(
            combined_table(pairs, entries, "Batch", "Other"), encoding="utf-8")

    # The group checkpoints are included so that the table shows both what
    # adaptation buys inside each backbone and that group normalisation has
    # no statistics to replace: only its source row is defined.
    (args.out / "cross_cohort.tex").write_text(
        cross_cohort_table([(args.cross / "eegnet_batch.json", "EEGNet"),
                            (args.cross / "eegnet_group.json", "EEGNet (group)"),
                            (args.cross / "sleepcnn_batch.json", "SleepCNN"),
                            (args.cross / "sleepcnn_group.json", "SleepCNN (group)")]),
        encoding="utf-8")

    for key, name in (("pooled|batch-vs-group", "sleep_edf_bn_vs_gn"),
                      ("pooled|batch-vs-adabn", "sleep_edf_bn_vs_adabn"),
                      ("pooled|group-vs-adabn", "sleep_edf_gn_vs_adabn")):
        (args.out / f"{name}.tex").write_text(
            pair_table(data["pairs"], key), encoding="utf-8")

    # Side-by-side per-backbone comparison, written as one tabular.
    (args.out / "sleep_edf_bn_vs_gn_both.tex").write_text(
        combined_table(data["pairs"],
                       [("eegnet|batch-vs-group", "EEGNet"),
                        ("sleepcnn|batch-vs-group", "SleepCNN")],
                       "Batch", "Group"),
        encoding="utf-8")
    (args.out / "sleep_edf_bn_vs_adabn_both.tex").write_text(
        combined_table(data["pairs"],
                       [("eegnet|batch-vs-adabn", "EEGNet"),
                        ("sleepcnn|batch-vs-adabn", "SleepCNN")],
                       "Batch", "Batch + AdaBN"),
        encoding="utf-8")
    print(f"wrote tables to {args.out}")


if __name__ == "__main__":
    main()
