"""Assemble the three arms of the normalisation comparison and test them.

The arms are

``batch``   the batch-normalised runs,
``group``   the group-normalised runs,
``adabn``   the batch-normalised checkpoints evaluated with recording
            statistics.

Every source is keyed by ``<model>/fold<f>_seed<s>`` so that runs are compared
only when they come from the same fold and the same seed, and therefore from
the same subjects, the same epochs and the same initialisation.

Example
-------
python scripts/aggregate_arms.py --out results/norm/arms.json
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
    "coverage", "set_size", "workload", "worst", "coverage_sd", "below",
    "probe", "probe_within",
    "centroid_ratio", "between_within_ratio", "swap_flip_rate",
]


def key_of(run: str) -> str | None:
    m = RUN_RE.search(Path(run).as_posix())
    return f"{m['model']}/fold{m['fold']}_seed{m['seed']}" if m else None


def per_subject_stats(method: dict) -> tuple[float, float]:
    values = np.asarray(list(method.get("per_subject_coverage", {}).values()), float)
    if values.size < 2:
        return float("nan"), float("nan")
    return float(values.std(ddof=1)), float((values < 0.9).mean())


def from_run_dir(run_dir: Path) -> dict:
    row: dict = {}
    match = RUN_RE.search(run_dir.as_posix())
    if match:
        row["model"], row["fold"], row["seed"] = match["model"], int(match["fold"]), int(match["seed"])
    for path in sorted(run_dir.glob("*_fold*_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        test = data.get("test_metrics") or data.get("test") or data
        for k in ("accuracy", "kappa", "mf1", "ece"):
            if test.get(k) is not None:
                row[k] = test[k]
    diag = run_dir / "cp_diagnostic.json"
    if diag.exists():
        method = json.loads(diag.read_text(encoding="utf-8"))["methods"].get("global", {})
        row["coverage"] = method.get("coverage")
        row["set_size"] = method.get("set_size")
        row["workload"] = method.get("review_workload")
        row["worst"] = method.get("worst_subject_coverage")
        row["coverage_sd"], row["below"] = per_subject_stats(method)
    return row


def collect(root: Path) -> dict[str, dict]:
    out = {}
    for run_dir in sorted(root.glob("*/fold*_seed*")):
        if not run_dir.is_dir():
            continue
        row = from_run_dir(run_dir)
        key = key_of(run_dir.as_posix())
        if key:
            out[key] = row
    return out


def collect_probes(root: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(root.glob("probe_*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
            key = key_of(entry["run"])
            if not key:
                continue
            within = [v for v in entry.get("within_stage", {}).values() if v is not None]
            out[key] = {"probe": entry.get("overall"),
                        "probe_within": float(np.mean(within)) if within else None,
                        "chance": entry.get("chance")}
    return out


def collect_subject_dependence(root: Path, prefix: Path) -> dict[str, dict]:
    """Read subject-dependence files, keeping only runs under ``prefix``."""
    out = {}
    if not root.exists():
        return out
    for path in sorted(root.glob("*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
            if not Path(entry["run"]).as_posix().startswith(prefix.as_posix()):
                continue
            key = key_of(entry["run"])
            if key:
                out[key] = {k: entry.get(k) for k in
                            ("centroid_ratio", "between_within_ratio", "swap_flip_rate")}
    return out


def collect_adabn(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for entry in data.get("runs", []):
        key = key_of(entry["run"])
        if not key:
            continue
        within = [v for v in entry.get("probe_within", {}).values() if v is not None]
        out[key] = {
            "model": key.split("/")[0],
            "fold": int(key.split("fold")[1].split("_")[0]),
            "seed": int(key.split("seed")[1]),
            "accuracy": entry.get("accuracy"),
            "kappa": entry.get("kappa"),
            "mf1": entry.get("mf1"),
            "ece": entry.get("ece"),
            "coverage": entry.get("coverage"),
            "set_size": entry.get("set_size"),
            "workload": entry.get("review_workload"),
            "worst": entry.get("worst_subject_coverage"),
            "coverage_sd": entry.get("coverage_sd"),
            "probe": entry.get("probe_overall"),
            "probe_within": float(np.mean(within)) if within else None,
            "per_subject_coverage": entry.get("per_subject_coverage", {}),
        }
        values = np.asarray(list(entry.get("per_subject_coverage", {}).values()), float)
        out[key]["below"] = float((values < 0.9).mean()) if values.size else None
    return out


def paired(a: dict[str, dict], b: dict[str, dict], keys: list[str]) -> dict:
    out = {}
    for metric in METRICS:
        pairs = [(a[k].get(metric), b[k].get(metric)) for k in keys
                 if a.get(k, {}).get(metric) is not None and b.get(k, {}).get(metric) is not None]
        if len(pairs) < 3:
            continue
        x = np.asarray([p[0] for p in pairs], float)
        y = np.asarray([p[1] for p in pairs], float)
        d = y - x
        sd = d.std(ddof=1)
        se = sd / np.sqrt(len(d))
        try:
            p_value = float(stats.wilcoxon(x, y).pvalue)
        except ValueError:
            p_value = float("nan")
        out[metric] = {
            "n": int(len(pairs)),
            "a_mean": float(x.mean()), "a_sd": float(x.std(ddof=1)),
            "b_mean": float(y.mean()), "b_sd": float(y.std(ddof=1)),
            "diff": float(d.mean()),
            "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "dz": float(d.mean() / sd) if sd > 0 else float("nan"),
            "wilcoxon_p": p_value,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, default=Path("results/main"))
    parser.add_argument("--group", type=Path, default=Path("results/probe_norm/group_all"))
    parser.add_argument("--probe-batch", type=Path, default=Path("results/main/summary"))
    parser.add_argument("--probe-group", type=Path,
                        default=Path("results/probe_norm/group_all/summary"))
    parser.add_argument("--subject-dep", type=Path, default=Path("results/norm/subject_dependence"))
    parser.add_argument("--adabn", type=Path, default=Path("results/norm"))
    parser.add_argument("--out", type=Path, default=Path("results/norm/arms.json"))
    args = parser.parse_args()

    arms = {
        "batch": collect(args.batch),
        "group": collect(args.group),
    }
    arms["batch"].update({k: {**arms["batch"].get(k, {}), **v}
                          for k, v in collect_probes(args.probe_batch).items()
                          if k in arms["batch"]})
    arms["group"].update({k: {**arms["group"].get(k, {}), **v}
                          for k, v in collect_probes(args.probe_group).items()
                          if k in arms["group"]})
    for arm, root in (("batch", args.batch), ("group", args.group)):
        for key, value in collect_subject_dependence(args.subject_dep, root).items():
            if key in arms[arm]:
                arms[arm][key].update(value)

    arms["adabn"] = {}
    for model in ("eegnet", "sleepcnn"):
        path = args.adabn / f"adabn_{model}.json"
        if path.exists():
            arms["adabn"].update(collect_adabn(path))
    # The subject-dependence numbers (centroid ratio, between/within ratio,
    # swap-flip rate) are properties of the representation, and AdaBN changes
    # the representation even though it leaves the weights alone: the batch
    # numbers do not transfer to the adapted arm. They are left unset for
    # AdaBN rather than copied, and neither ``summary`` nor ``paired`` uses
    # them when they are ``None``. Paper 2 does not report them for AdaBN.
    for key in arms["adabn"]:
        for metric in ("centroid_ratio", "between_within_ratio", "swap_flip_rate"):
            arms["adabn"][key].pop(metric, None)

    result: dict = {"arms": arms, "pairs": {}, "summary": {}}
    for model in sorted({k.split("/")[0] for k in arms["batch"]}):
        keys = sorted(k for k in arms["batch"] if k.startswith(model + "/"))
        for a, b in (("batch", "group"), ("batch", "adabn"), ("group", "adabn")):
            common = [k for k in keys if k in arms[a] and k in arms[b]]
            if len(common) >= 3:
                result["pairs"][f"{model}|{a}-vs-{b}"] = paired(arms[a], arms[b], common)
    all_keys = sorted(set.intersection(*[set(v) for v in arms.values()]))
    for a, b in (("batch", "group"), ("batch", "adabn"), ("group", "adabn")):
        if all_keys:
            result["pairs"][f"pooled|{a}-vs-{b}"] = paired(arms[a], arms[b], all_keys)

    for name, arm in arms.items():
        result["summary"][name] = {}
        for model in sorted({k.split("/")[0] for k in arm}):
            rows = [v for k, v in arm.items() if k.startswith(model + "/")]
            result["summary"][name][model] = {
                metric: float(np.mean([r[metric] for r in rows if r.get(metric) is not None]))
                for metric in METRICS
                if any(r.get(metric) is not None for r in rows)
            }
            result["summary"][name][model]["n_runs"] = len(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    print(f"{'comparison':28s} {'metric':18s} {'a':>7s} {'b':>7s} {'diff':>8s} "
          f"{'dz':>6s} {'p':>9s} {'n':>3s}")
    for name, stats_ in result["pairs"].items():
        for metric in ("accuracy", "set_size", "worst", "coverage_sd", "probe"):
            if metric in stats_:
                s = stats_[metric]
                print(f"{name:28s} {metric:18s} {s['a_mean']:7.3f} {s['b_mean']:7.3f} "
                      f"{s['diff']:+8.3f} {s['dz']:+6.2f} {s['wilcoxon_p']:9.4f} {s['n']:3d}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
