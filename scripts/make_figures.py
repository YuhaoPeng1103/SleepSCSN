"""Generate the figures of the paper from ``results/merged``.

The arms are ordered by how much of the recording they condition on, so the
colour scale is ordinal: grey for the uncorrected model, aqua for the
post-hoc marginal correction, orange for layerwise AdaBN and blue for the
stage-conditional correction the paper proposes.  Line styles repeat the
ordering so that the figures remain readable in greyscale.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("paper2/figures")
FIG.mkdir(parents=True, exist_ok=True)

STAGES = ["W", "N1", "N2", "N3", "REM"]
ARMS = [
    ("bn", "BN", "#8a8a86", ":", "o"),
    ("marginal", "Marginal", "#1baf7a", "-.", "s"),
    ("adabn_layerwise", "AdaBN", "#eb6834", "--", "^"),
    ("cond_full", "SCSN", "#2a78d6", "-", "D"),
]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a46",
    "axes.labelcolor": "#222220",
    "text.color": "#222220",
    "xtick.color": "#4a4a46",
    "ytick.color": "#4a4a46",
    "grid.color": "#d8d8d2",
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})


def load(key: str) -> dict:
    path = Path("results/merged") / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def runs(d: dict, arm: str) -> list:
    return [r["results"][arm] for r in d["runs"] if arm in r["results"]]


def stage_matrix(d: dict, arm: str) -> np.ndarray:
    rows = []
    for r in d["runs"]:
        if arm not in r["results"]:
            continue
        pw = r["results"][arm].get("probe_within") or {}
        rows.append([np.nan if pw.get(s) is None else pw[s] for s in STAGES])
    return np.asarray(rows, dtype=float)


def fig_stage_identity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    for ax, key, title in zip(axes, ["edf_bn_eegnet", "hmc_bn_eegnet"],
                              ["Sleep-EDF (15 runs)", "HMC (10 runs)"]):
        d = load(key)
        present = [(a, l, c, ls, m) for a, l, c, ls, m in ARMS
                   if runs(d, a)]
        width = 0.8 / max(len(present), 1)
        x = np.arange(len(STAGES))
        for i, (arm, label, color, ls, marker) in enumerate(present):
            mats = stage_matrix(d, arm)
            if mats.size == 0:
                continue
            mean = np.nanmean(mats, axis=0)
            count = np.sum(~np.isnan(mats), axis=0)
            se = np.nanstd(mats, axis=0, ddof=1) / np.sqrt(np.maximum(count, 1))
            ax.bar(x + (i - (len(present) - 1) / 2) * width, mean, width * 0.92,
                   color=color, label=label, yerr=se, capsize=1.5,
                   error_kw={"linewidth": 0.6, "ecolor": "#4a4a46"})
        chance = 1 / 4 if key.startswith("edf") else 1 / 31
        ax.axhline(chance, color="#4a4a46", lw=0.7, ls=(0, (2, 3)))
        ax.set_xticks(x, STAGES)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Subject identification (within stage)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.6)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.05, -0.16),
                   frameon=False, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(FIG / "stage_identity.pdf", bbox_inches="tight")
    fig.savefig(FIG / "stage_identity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_paired_probe() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    for ax, key, title in zip(axes, ["edf_bn_eegnet", "hmc_bn_eegnet"],
                              ["Sleep-EDF, EEGNet", "HMC, EEGNet"]):
        d = load(key)
        present = [(a, l, c, m) for a, l, c, _, m in ARMS if runs(d, a)]
        x = np.arange(len(present))
        for r in d["runs"]:
            ys = [r["results"][a]["probe_within_mean"] for a, *_ in present
                  if a in r["results"] and
                  r["results"][a].get("probe_within_mean") is not None]
            if len(ys) == len(present):
                ax.plot(x, ys, color="#b8b8b2", lw=0.7, zorder=1)
        for i, (arm, label, color, marker) in enumerate(present):
            vals = [r["results"][arm]["probe_within_mean"] for r in d["runs"]
                    if arm in r["results"] and
                    r["results"][arm].get("probe_within_mean") is not None]
            ax.scatter([i] * len(vals), vals, s=14, color=color, marker=marker,
                       zorder=3, edgecolor="white", linewidth=0.4)
            ax.plot([i - 0.18, i + 0.18], [np.mean(vals)] * 2, color=color,
                    lw=2.0, zorder=4)
        chance = 1 / 4 if key.startswith("edf") else 1 / 31
        ax.axhline(chance, color="#4a4a46", lw=0.7, ls=(0, (2, 3)))
        ax.set_xticks(x, [l for _, l, _, _ in present])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Within-stage subject probe")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "paired_probe.pdf", bbox_inches="tight")
    fig.savefig(FIG / "paired_probe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def curve(d: dict, arm: str, metric: str):
    xs, ys = [], []
    for r in d["runs"]:
        if arm in r["results"] and r["results"][arm].get(metric) is not None:
            xs.append(arm)
            ys.append(r["results"][arm][metric])
    return np.asarray(ys)


def fig_interpolation() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    lam = [0.25, 0.5, 0.75, 1.0]
    for col, key in enumerate(["edf_bn_eegnet", "hmc_bn_eegnet"]):
        d = load(key)
        for row, metric in enumerate(["probe_within_mean", "accuracy"]):
            ax = axes[row, col]
            for arm_list, label, color in [
                (["cond_full_lam0.25", "cond_full_lam0.5",
                  "cond_full_lam0.75", "cond_full"], "SCSN, $\\lambda$ path",
                 "#2a78d6"),
            ]:
                means, ses = [], []
                for a in arm_list:
                    v = curve(d, a, metric)
                    means.append(v.mean() if len(v) else np.nan)
                    ses.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                ax.errorbar(lam, means, yerr=ses, color=color, marker="o",
                            lw=1.4, ms=3, capsize=2, label=label)
            ax.set_xticks(lam)
            ax.set_xlabel("$\\lambda$")
            ax.set_ylabel("Within-stage probe" if row == 0 else "Accuracy")
            ax.grid(alpha=0.5)
            if row == 0:
                ax.set_title("Sleep-EDF, EEGNet" if col == 0 else "HMC, EEGNet")
    fig.tight_layout()
    fig.savefig(FIG / "interpolation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "interpolation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_rank() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ranks = [1, 2, 4]
    for ax, metric, ylabel in zip(axes,
                                  ["probe_within_mean", "accuracy"],
                                  ["Within-stage probe", "Accuracy"]):
        for key, label, color, marker in [
            ("edf_bn_eegnet", "Sleep-EDF", "#2a78d6", "o"),
            ("hmc_bn_eegnet", "HMC", "#eb6834", "s"),
        ]:
            d = load(key)
            means, ses = [], []
            for r in ranks:
                v = curve(d, f"cond_r{r}", metric)
                means.append(v.mean() if len(v) else np.nan)
                ses.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            ax.errorbar(ranks, means, yerr=ses, color=color, marker=marker,
                        lw=1.4, ms=3, capsize=2, label=label)
        ax.set_xticks(ranks)
        ax.set_xlabel("Correction rank $r$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.5)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rank.pdf", bbox_inches="tight")
    fig.savefig(FIG / "rank.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_gn() -> None:
    keys = ["edf_gn_eegnet", "edf_gn_sleepcnn", "hmc_gn_eegnet", "hmc_gn_sleepcnn"]
    labels = ["EDF\nEEGNet", "EDF\nSleepCNN", "HMC\nEEGNet", "HMC\nSleepCNN"]
    arms = [("bn", "GN, uncorrected", "#8a8a86"),
            ("marginal", "GN + marginal", "#1baf7a"),
            ("cond_full", "GN + SCSN", "#2a78d6")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, metric, ylabel in zip(axes, ["accuracy", "probe_within_mean"],
                                  ["Accuracy", "Within-stage probe"]):
        width = 0.26
        x = np.arange(len(keys))
        for i, (arm, label, color) in enumerate(arms):
            means, ses = [], []
            for key in keys:
                d = load(key) if (Path("results/merged") / f"{key}.json").exists() else {"runs": []}
                v = curve(d, arm, metric)
                means.append(v.mean() if len(v) else np.nan)
                ses.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            ax.bar(x + (i - 1) * width, means, width * 0.92, color=color,
                   label=label, yerr=ses, capsize=1.5,
                   error_kw={"linewidth": 0.6, "ecolor": "#4a4a46"})
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.6)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.05, -0.18),
                   frameon=False, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(FIG / "gn_arms.pdf", bbox_inches="tight")
    fig.savefig(FIG / "gn_arms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_coverage() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, key, title in zip(axes, ["edf_bn_eegnet", "hmc_bn_eegnet"],
                              ["Sleep-EDF", "HMC"]):
        d = load(key)
        for arm, label, color, ls, _ in ARMS:
            vals = []
            for r in d["runs"]:
                if arm not in r["results"]:
                    continue
                pc = r["results"][arm].get("per_subject_coverage") or {}
                vals.extend(v for v in pc.values() if v is not None)
            if not vals:
                continue
            vals = np.sort(np.asarray(vals))
            ecdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, ecdf, where="post", color=color, ls=ls, lw=1.3,
                    label=label)
        ax.axvline(0.9, color="#4a4a46", lw=0.7, ls=(0, (2, 3)))
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Per-patient coverage at $\\alpha=0.1$")
        ax.set_title(title)
        ax.grid(alpha=0.5)
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "coverage_ecdf.pdf", bbox_inches="tight")
    fig.savefig(FIG / "coverage_ecdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_cross() -> None:
    paths = sorted(Path("results/scsn_cross").glob("*.json"))
    if not paths:
        return
    data = {}
    for p in paths:
        data[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    arms = [("cohort", "Cohort"), ("adabn", "Marginal"), ("scsn", "SCSN"),
            ("adabn_layerwise", "AdaBN"), ("ada_scsn", "AdaBN+SCSN"),
            ("scsn_oracle", "SCSN oracle")]
    colors = ["#8a8a86", "#1baf7a", "#2a78d6", "#eb6834", "#6250d6", "#b8b8b2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, metric, ylabel in zip(axes, ["accuracy", "recording_accuracy_worst"],
                                  ["Accuracy", "Worst recording accuracy"]):
        x = np.arange(len(arms))
        for j, (model, d) in enumerate(data.items()):
            means = [d[a][metric] if a in d else np.nan for a, _ in arms]
            ax.bar(x + (j - (len(data) - 1) / 2) * 0.36, means, 0.34,
                   color=["#2a78d6", "#eb6834"][j % 2],
                   label=model.title(), alpha=0.9)
        ax.set_xticks(x, [l for _, l in arms], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "cross_cohort.pdf", bbox_inches="tight")
    fig.savefig(FIG / "cross_cohort.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_stage_identity()
    fig_paired_probe()
    fig_interpolation()
    fig_rank()
    fig_gn()
    fig_coverage()
    fig_cross()
    print("wrote figures to", FIG)
