"""Emit the LaTeX tables of the paper from ``results/final_stats.json``.

Every table is written to ``paper2/tex/tables``; missing arms are printed as
an em dash so that a partially finished matrix still compiles.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("results")
OUT = Path("paper2/tex/tables")
OUT.mkdir(parents=True, exist_ok=True)

ARMS = [
    ("bn", "BN"),
    ("marginal", "Marginal"),
    ("adabn_layerwise", "AdaBN"),
    ("cond_full", "SCSN"),
    ("cond_full_lam0.5", "SCSN-$\\lambda_{0.5}$"),
    ("tent_sgd", "Tent-SGD"),
]


def fmt(v, nd=3):
    if v is None:
        return "--"
    return f"{v:.{nd}f}"


def get(ds, arm, metric, nd=3):
    a = ds.get("arms", {}).get(arm)
    if not a or a.get(metric) is None:
        return "--"
    return fmt(a[metric], nd)


def main_table(path: Path, arch: str, datasets: list[tuple[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Cohort & Arm & $n$ & Acc & $\kappa$ & MF1 & ECE & Probe & Within & Worst & SD \\",
        r"\midrule",
    ]
    for i, (key, label) in enumerate(datasets):
        ds = stats["datasets"].get(key, {"arms": {}})
        if i:
            lines.append(r"\addlinespace")
        first = True
        for arm, arm_label in ARMS:
            if arm not in ("bn", "marginal", "cond_full", "cond_full_lam0.5") and \
                    arch == "sleepcnn" and arm == "tent_sgd":
                pass
            if key.endswith("_gn") and arm in ("adabn_layerwise", "tent_sgd"):
                continue
            n = ds.get("arms", {}).get(arm, {}).get("n")
            name = label if first else ""
            first = False
            lines.append(
                f"{name} & {arm_label} & {n if n else '--'} & "
                f"{get(ds, arm, 'accuracy')} & {get(ds, arm, 'kappa')} & "
                f"{get(ds, arm, 'mf1')} & {get(ds, arm, 'ece')} & "
                f"{get(ds, arm, 'probe')} & "
                f"{get(ds, arm, 'probe_within_mean')} & "
                f"{get(ds, arm, 'worst_subject_coverage')} & "
                f"{get(ds, arm, 'coverage_sd')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def controls_table() -> None:
    rows = [
        ("bn", "BN"),
        ("marginal", "Marginal"),
        (r"cond_full_lam0.25", r"SCSN-$\lambda_{0.25}$"),
        (r"cond_full_lam0.5", r"SCSN-$\lambda_{0.5}$"),
        (r"cond_full_lam0.75", r"SCSN-$\lambda_{0.75}$"),
        ("cond_full", "SCSN ($\\lambda=1$)"),
        ("cond_r1", "SCSN-$r_1$"),
        ("cond_r2", "SCSN-$r_2$"),
        ("cond_r4", "SCSN-$r_4$"),
        ("cond_r4_orth", "SCSN-orth"),
        ("cond_hard", "SCSN-hard"),
        ("cond_mismatch", "SCSN-mismatch"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Sleep-EDF, EEGNet} & \multicolumn{4}{c}{HMC, EEGNet} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Arm & Acc & Within & Worst & SD & Acc & Within & Worst & SD \\",
        r"\midrule",
    ]
    for arm, label in rows:
        cells = []
        for key in ("edf_bn_eegnet", "hmc_bn_eegnet"):
            ds = stats["datasets"].get(key, {"arms": {}})
            cells += [get(ds, arm, "accuracy"), get(ds, arm, "probe_within_mean"),
                      get(ds, arm, "worst_subject_coverage"),
                      get(ds, arm, "coverage_sd")]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "controls.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pairs_table() -> None:
    contrasts = [
        ("cond_full", "bn", "SCSN $-$ BN"),
        ("cond_full", "marginal", "SCSN $-$ Marginal"),
        ("cond_full", "adabn_layerwise", "SCSN $-$ AdaBN"),
        ("adabn_layerwise", "bn", "AdaBN $-$ BN"),
        ("marginal", "bn", "Marginal $-$ BN"),
    ]
    keys = ["edf_bn_eegnet", "edf_bn_sleepcnn", "hmc_bn_eegnet", "hmc_bn_sleepcnn"]
    head = " & ".join("\\shortstack{" + k.replace("_", "\\\\") + "}" for k in keys)
    lines = [
        r"\begin{tabular}{l" + "c" * len(keys) + "}",
        r"\toprule",
        r"Contrast & " + head + r" \\",
        r"\midrule",
        r"\multicolumn{" + str(len(keys) + 1) + r"}{l}{\emph{Accuracy difference}} \\",
    ]
    for a, b, label in contrasts:
        cells = []
        for key in keys:
            p = stats["pairs"].get(f"{key}:{a}_vs_{b}")
            if not p or not p.get("accuracy"):
                cells.append("--")
            else:
                q = p["accuracy"]
                cells.append(f"{q['diff']:+.3f} ({q['p']:.3g})")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{" + str(len(keys) + 1) + r"}{l}{\emph{Within-stage probe difference}} \\")
    for a, b, label in contrasts:
        cells = []
        for key in keys:
            p = stats["pairs"].get(f"{key}:{a}_vs_{b}")
            if not p or not p.get("probe_within_mean"):
                cells.append("--")
            else:
                q = p["probe_within_mean"]
                cells.append(f"{q['diff']:+.3f} ({q['p']:.3g})")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "pairs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cross_table() -> None:
    rows = [
        ("cohort", "Cohort statistics"),
        ("adabn", "Marginal"),
        ("scsn", "SCSN"),
        ("scsn_lam0.5", "SCSN-$\\lambda_{0.5}$"),
        ("adabn_layerwise", "AdaBN"),
        ("ada_scsn", "AdaBN + SCSN"),
        ("scsn_oracle", "SCSN (oracle labels)"),
    ]
    paths = sorted(Path("results/scsn_cross").glob("*.json"))
    if not paths:
        return
    data = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths}
    models = list(data)
    lines = [
        r"\begin{tabular}{l" + "r" * (3 * len(models)) + "}",
        r"\toprule",
    ]
    labels = {"eegnet": "EEGNet", "sleepcnn": "SleepCNN"}
    head = ""
    for m in models:
        head += r" & \multicolumn{3}{c}{" + labels.get(m, m.replace("_", " ").title()) + "}"
    lines.append(head + r" \\")
    lines.append(r"\cmidrule(lr){2-4}" * len(models))
    sub = ""
    for _ in models:
        sub += r" & Acc & Worst & SD"
    lines.append(sub + r" \\")
    lines.append(r"\midrule")
    for arm, label in rows:
        cells = []
        for m in models:
            d = data[m]
            if arm not in d:
                cells += ["--", "--", "--"]
                continue
            r = d[arm]
            cells += [fmt(r.get("accuracy")),
                      fmt(r.get("recording_accuracy_worst")),
                      fmt(r.get("recording_accuracy_sd"))]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "cross_cohort.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    stats = json.loads((ROOT / "final_stats.json").read_text(encoding="utf-8"))
    main_table(OUT / "main_eegnet.tex", "eegnet",
               [("edf_bn_eegnet", "Sleep-EDF BN"),
                ("hmc_bn_eegnet", "HMC BN"),
                ("edf_gn_eegnet", "Sleep-EDF GN"),
                ("hmc_gn_eegnet", "HMC GN")])
    main_table(OUT / "main_sleepcnn.tex", "sleepcnn",
               [("edf_bn_sleepcnn", "Sleep-EDF BN"),
                ("hmc_bn_sleepcnn", "HMC BN"),
                ("edf_gn_sleepcnn", "Sleep-EDF GN"),
                ("hmc_gn_sleepcnn", "HMC GN")])
    controls_table()
    pairs_table()
    cross_table()
    print("wrote tables to", OUT)
