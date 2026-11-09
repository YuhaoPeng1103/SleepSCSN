"""Pair batch-normalised and group-normalised runs and compare them.

Both roots are expected to hold the same run layout,
``<root>/<model>/fold<f>_seed<s>/``, with the training JSON, ``cp_diagnostic``
and ``temporal_cp`` analyses. Probe and subject-dependence summaries are merged
in when their JSON files exist.

The output is one row per run plus paired statistics (mean difference,
Wilcoxon signed-rank p-value and Cohen's d_z) for each metric, per backbone and
pooled over backbones.

Example
-------
python scripts/aggregate_norm.py \
    --bn results/main --gn results/probe_norm/group_all \
    --probe-bn results/main/summary --probe-gn results/probe_norm/group_all/summary \
    --subject-dep results/norm/subject_dependence \
    --out results/norm/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

RUN_RE = re.compile(r"(?P<model>[a-z]+)/fold(?P<fold>\d+)_seed(?P<seed>\d+)$")

METRICS = [
    "accuracy", "kappa", "mf1", "ece",
    "cov_global", "set_global", "workload_global", "worst_global",
    "sd_global", "below_global",
    "cov_deficit", "set_deficit", "workload_deficit", "worst_deficit",
    "sd_deficit", "below_deficit",
    "probe_overall", "probe_within",
    "centroid_ratio", "between_within_ratio", "swap_flip_rate", "swap_shift",
]


def _per_subject_stats(method: dict) -> tuple[float, float]:
    values = np.asarray(list(method.get("per_subject_coverage", {}).values()), dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    return float(values.std(ddof=1)), float((values < 0.9).mean())


def load_run(run_dir: Path) -> dict:
    row: dict = {"run": str(run_dir)}
    match = RUN_RE.search(run_dir.as_posix())
    if match:
        row.update(match.groupdict())
        row["fold"] = int(row["fold"])
        row["seed"] = int(row["seed"])

    train = sorted(run_dir.glob("*_fold*_seed*.json"))
    train = [p for p in train if p.name.startswith(row.get("model", ""))]
    if train:
        data = json.loads(train[0].read_text(encoding="utf-8"))
        test = data.get("test_metrics") or data.get("test") or data
        for key in ("accuracy", "kappa", "mf1", "ece"):
            row[key] = test.get(key)

    diag = run_dir / "cp_diagnostic.json"
    if diag.exists():
        methods = json.loads(diag.read_text(encoding="utf-8"))["methods"]
        method = methods.get("global")
        if method:
            row["cov_global"] = method.get("coverage")
            row["set_global"] = method.get("set_size")
            row["workload_global"] = method.get("review_workload")
            row["worst_global"] = method.get("worst_subject_coverage")
            sd, below = _per_subject_stats(method)
            row["sd_global"], row["below_global"] = sd, below

    temporal = run_dir / "temporal_cp.json"
    if temporal.exists():
        methods = json.loads(temporal.read_text(encoding="utf-8"))["methods"]
        method = methods.get("per_epoch_deficit__all_epochs") or {}
        row["cov_deficit"] = method.get("coverage")
        row["set_deficit"] = method.get("set_size")
        row["workload_deficit"] = method.get("review_workload")
        row["worst_deficit"] = method.get("worst_subject_coverage")
        sd, below = _per_subject_stats(method)
        row["sd_deficit"], row["below_deficit"] = sd, below
    return row


def merge_probes(rows: dict[str, dict], summary_dir: Path) -> None:
    for path in sorted(summary_dir.glob("probe_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("runs", []):
            key = Path(entry["run"]).as_posix()
            if key in rows:
                rows[key]["probe_overall"] = entry.get("overall")
                within = [v for v in entry.get("within_stage", {}).values() if v is not None]
                rows[key]["probe_within"] = float(np.mean(within)) if within else None


def merge_subject_dependence(rows: dict[str, dict], root: Path) -> None:
    if not root or not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("runs", []):
            key = Path(entry["run"]).as_posix()
            if key in rows:
                for metric in ("centroid_ratio", "between_within_ratio",
                               "swap_flip_rate", "swap_shift"):
                    rows[key][metric] = entry.get(metric)


def paired_stats(bn: list[dict], gn: list[dict]) -> dict:
    out = {}
    for metric in METRICS:
        pairs = [(b.get(metric), g.get(metric)) for b, g in zip(bn, gn)
                 if b.get(metric) is not None and g.get(metric) is not None]
        if len(pairs) < 3:
            continue
        b = np.asarray([p[0] for p in pairs], dtype=float)
        g = np.asarray([p[1] for p in pairs], dtype=float)
        diff = g - b
        sd = diff.std(ddof=1)
        try:
            p_value = float(stats.wilcoxon(b, g).pvalue)
        except ValueError:
            p_value = float("nan")
        out[metric] = {
            "n": int(len(pairs)),
            "bn_mean": float(b.mean()),
            "bn_sd": float(b.std(ddof=1)),
            "gn_mean": float(g.mean()),
            "gn_sd": float(g.std(ddof=1)),
            "diff": float(diff.mean()),
            "diff_ci95": [float(diff.mean() - 1.96 * sd / np.sqrt(len(diff))),
                          float(diff.mean() + 1.96 * sd / np.sqrt(len(diff)))],
            "dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
            "wilcoxon_p": p_value,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bn", type=Path, default=Path("results/main"))
    parser.add_argument("--gn", type=Path, default=Path("results/probe_norm/group_all"))
    parser.add_argument("--probe-bn", type=Path, default=None)
    parser.add_argument("--probe-gn", type=Path, default=None)
    parser.add_argument("--subject-dep", type=Path, default=Path("results/norm/subject_dependence"))
    parser.add_argument("--out", type=Path, default=Path("results/norm/summary.json"))
    args = parser.parse_args()

    rows = {}
    for root in (args.bn, args.gn):
        for run_dir in sorted(root.glob("*/fold*_seed*")):
            if not run_dir.is_dir():
                continue
            row = load_run(run_dir)
            rows[row["run"]] = row

    for root, probe in ((args.bn, args.probe_bn), (args.gn, args.probe_gn)):
        if probe:
            sub = {k: v for k, v in rows.items() if k.startswith(root.as_posix())}
            merge_probes(sub, probe)
            rows.update(sub)
    if args.subject_dep:
        for path in sorted(args.subject_dep.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("runs", []):
                key = Path(entry["run"]).as_posix()
                if key in rows:
                    for metric in ("centroid_ratio", "between_within_ratio",
                                   "swap_flip_rate", "swap_shift"):
                        rows[key][metric] = entry.get(metric)

    def select(root: Path) -> dict[str, dict]:
        return {k: v for k, v in rows.items() if k.startswith(root.as_posix())}

    bn_rows, gn_rows = select(args.bn), select(args.gn)
    key_of = lambda r: f"{r['model']}/fold{r['fold']}_seed{r['seed']}"

    results: dict = {"per_model": {}, "pooled": {}, "runs": {"bn": list(bn_rows.values()),
                                                            "gn": list(gn_rows.values())}}
    for model in sorted({r["model"] for r in bn_rows.values()}):
        bn_m = [r for r in bn_rows.values() if r["model"] == model]
        gn_m = {key_of(r): r for r in gn_rows.values() if r["model"] == model}
        bn_m = [r for r in bn_m if key_of(r) in gn_m]
        gn_m = [gn_m[key_of(r)] for r in bn_m]
        if bn_m:
            results["per_model"][model] = paired_stats(bn_m, gn_m)

    bn_sorted = {key_of(r): r for r in bn_rows.values()}
    gn_sorted = {key_of(r): r for r in gn_rows.values()}
    keys = sorted(set(bn_sorted) & set(gn_sorted))
    results["pooled"] = paired_stats([bn_sorted[k] for k in keys],
                                     [gn_sorted[k] for k in keys])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"{'metric':22s} {'BN':>8s} {'GN':>8s} {'diff':>8s} {'dz':>6s} {'p':>9s}")
    for metric, s in results["pooled"].items():
        print(f"{metric:22s} {s['bn_mean']:8.3f} {s['gn_mean']:8.3f} "
              f"{s['diff']:+8.3f} {s['dz']:+6.2f} {s['wilcoxon_p']:9.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
