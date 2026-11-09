"""Figures for the normalisation paper.

Figure 1  per-patient conformal coverage and the subject probe, by arm.
Figure 2  patient identification as a function of depth (needs layerwise
          summaries).
Figure 3  set size against worst-patient coverage, one marker per run.
Figure 4  per-recording conformal coverage on the external cohort (HMC).

Example
-------
python scripts/make_figures2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [("batch", "Batch (cohort)", "batch_normalised"),
        ("group", "Group (per sample)", "group_normalised"),
        ("adabn", "Batch + AdaBN", "batch_adabn")]
COLORS = {"batch": "#0072B2", "group": "#E69F00", "adabn": "#009E73"}
ALPHA = 0.1

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def per_patient_coverage(root: Path, model: str) -> list[float]:
    values = []
    for path in sorted(root.glob(f"{model}/fold*_seed*/cp_diagnostic.json")):
        method = json.loads(path.read_text(encoding="utf-8"))["methods"]["global"]
        values.extend(float(v) for v in method.get("per_subject_coverage", {}).values())
    return values


def adabn_coverage(path: Path) -> list[float]:
    values = []
    for run in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
        values.extend(float(v) for v in run.get("per_subject_coverage", {}).values())
    return values


def probes(root: Path, model: str) -> list[float]:
    path = root / "summary" / f"probe_{model}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r["overall"] for r in data.get("runs", []) if r.get("overall")]


def adabn_probes(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r["probe_overall"] for r in data.get("runs", []) if r.get("probe_overall")]


def figure_coverage(cov: dict[str, dict[str, list[float]]], out: Path,
                    model: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    for i, (key, label, _) in enumerate(ARMS):
        v = np.sort(np.asarray(cov[key][model]))
        if v.size == 0:
            continue
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post",
                color=COLORS[key], lw=1.4, label=label)
    ax.axvline(1 - ALPHA, color="0.35", ls=(0, (3, 2)), lw=0.9)
    ax.text(1 - ALPHA, 0.04, " nominal", color="0.35", fontsize=7, va="bottom")
    ax.set_xlabel("Per-patient coverage")
    ax.set_ylabel("Fraction of patients")
    ax.set_xlim(0.68, 1.005)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("(a) Patient-level coverage", loc="left", fontsize=9)

    ax = axes[1]
    for i, (key, label, _) in enumerate(ARMS):
        v = cov[f"probe_{key}"][model]
        if not v:
            continue
        jitter = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.16
        ax.scatter(np.full(len(v), i) + jitter, v, s=16, color=COLORS[key],
                   alpha=0.85, linewidths=0)
        ax.hlines(np.mean(v), i - 0.24, i + 0.24, color=COLORS[key], lw=2)
    ax.axhline(0.25, color="0.35", ls=(0, (3, 2)), lw=0.9)
    ax.text(-0.42, 0.262, "chance", color="0.35", fontsize=7)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(["Batch", "Group", "Batch\n+AdaBN"], fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.15, 0.9)
    ax.set_ylabel("Patient identification (probe)")
    ax.set_title("(b) Decodable identity", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_hmc_coverage(batch_root: Path, adabn_path: Path, out: Path,
                        model: str = "eegnet") -> None:
    """Per-recording conformal coverage on the external cohort."""
    batch = per_patient_coverage(batch_root, model)
    adabn = adabn_coverage(adabn_path) if adabn_path.exists() else []
    if not batch:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for values, key, label in ((batch, "batch", "Batch (cohort)"),
                               (adabn, "adabn", "Batch + AdaBN")):
        v = np.sort(np.asarray(values))
        if v.size == 0:
            continue
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post",
                color=COLORS[key], lw=1.4, label=label)
    ax.axvline(1 - ALPHA, color="0.35", ls=(0, (3, 2)), lw=0.9)
    ax.text(1 - ALPHA, 0.03, " nominal", color="0.35", fontsize=7, va="bottom")
    ax.set_xlabel("Per-recording coverage")
    ax.set_ylabel("Fraction of recordings")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("HMC, EEGNet", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_pareto(arms: dict, out: Path, model: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for key, label, _ in ARMS:
        rows = [r for r in arms.get(key, {}).values() if r.get("model") == model]
        x = [r["worst"] for r in rows if r.get("worst") is not None]
        y = [r["set_size"] for r in rows if r.get("set_size") is not None]
        if not x:
            continue
        ax.scatter(x, y, s=14, color=COLORS[key], alpha=0.75, linewidths=0,
                   label=label)
    ax.set_xlabel("Worst-patient coverage")
    ax.set_ylabel("Mean set size")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Reliability against efficiency", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def layerwise_curve(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", {})


def figure_layerwise(summ: dict[str, dict[str, dict]], out: Path,
                     model: str) -> None:
    """Patient identification against depth, raw and after removing the
    per-recording mean."""
    panels = [(k, lbl) for k, lbl in (("main", "Batch (cohort)"),
                                      ("group_all", "Group (per sample)"))
              if summ.get(k)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.8),
                             sharey=True, squeeze=False)
    for ax, (key, label) in zip(axes[0], panels):
        layers = summ[key]
        names = [n for n in layers if n != "acc" and n != "acc_centered"
                 and n != "n_runs" and n != "chance"]
        x = np.arange(len(names))
        raw = [layers[n]["epoch_probe"] for n in names]
        cen = [layers[n]["centered_probe"] for n in names]
        chance = layers.get("chance")
        color = COLORS["batch" if key == "main" else "group"]
        ax.plot(x, raw, "-o", color=color, ms=4.5, lw=1.4,
                label="Pooled activation")
        ax.plot(x, cen, "--o", color=color, ms=4, lw=1.2, mfc="white",
                label="Recording mean removed")
        if chance:
            ax.axhline(chance, color="0.35", ls=(0, (3, 2)), lw=0.9)
            ax.text(0.02, chance + 0.02, "chance", color="0.35", fontsize=7,
                    transform=ax.get_yaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
        ax.set_xlabel("Depth")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{label}", loc="left", fontsize=8.5)
        ax.legend(fontsize=7, loc="lower left")
    axes[0][0].set_ylabel("Patient identification (probe)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, default=Path("results/main"))
    parser.add_argument("--group", type=Path, default=Path("results/probe_norm/group_all"))
    parser.add_argument("--norm", type=Path, default=Path("results/norm"))
    parser.add_argument("--out", type=Path, default=Path("paper2/figures"))
    parser.add_argument("--model", default="eegnet")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cov: dict = {
        "batch": {}, "group": {}, "adabn": {},
        "probe_batch": {}, "probe_group": {}, "probe_adabn": {},
    }
    for model in ("eegnet", "sleepcnn"):
        cov["batch"][model] = per_patient_coverage(args.batch, model)
        cov["group"][model] = per_patient_coverage(args.group, model)
        adabn_path = args.norm / f"adabn_{model}.json"
        cov["adabn"][model] = adabn_coverage(adabn_path) if adabn_path.exists() else []
        cov["probe_batch"][model] = probes(args.batch, model)
        cov["probe_group"][model] = probes(args.group, model)
        cov["probe_adabn"][model] = adabn_probes(adabn_path) if adabn_path.exists() else []

    figure_coverage(cov, args.out / f"coverage_probe_{args.model}.pdf", args.model)

    summ = {
        "main": layerwise_curve(args.norm / "layerwise" / f"main_{args.model}.json"),
        "group_all": layerwise_curve(
            args.norm / "layerwise" / f"group_all_{args.model}.json"),
    }
    figure_layerwise(summ, args.out / "layerwise_identity.pdf", args.model)

    arms_path = args.norm / "arms.json"
    if arms_path.exists():
        arms = json.loads(arms_path.read_text(encoding="utf-8"))["arms"]
        figure_pareto(arms, args.out / f"pareto_{args.model}.pdf", args.model)

    hmc_root = args.norm.parent / "hmc" / "batch"
    hmc_adabn = args.norm / "adabn_hmc_eegnet.json"
    if hmc_root.exists() and hmc_adabn.exists():
        figure_hmc_coverage(hmc_root, hmc_adabn, args.out / "hmc_coverage.pdf")
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
