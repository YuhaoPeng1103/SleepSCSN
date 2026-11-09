"""Why post-hoc marginal normalisation can hurt on a large cohort.

For one run, compute for every test recording

- the true sleep-stage distribution and its Jensen-Shannon divergence from the
  training cohort's stage distribution;
- the stage accuracy with no correction and after subtracting the recording's
  global feature-mean shift (post-hoc AdaBN);
- the paired accuracy change.

Then correlate the accuracy change with the divergence. The hypothesis is that
a recording whose stage mix is atypical has a global feature mean pulled toward
its dominant stage; subtracting that mean moves every epoch away from the
dominant stage and costs accuracy. The stage-conditional correction, which
never touches the between-stage offsets, should not have this failure mode.

Example
-------
python scripts/analyze_marginal_failure.py \
    --cache results/pilot_cssn/hmc_eegnet_fold0_seed0_features.npz \
    --run results/hmc/batch/eegnet/fold0_seed0/eegnet_fold0_seed0.pt \
    --out results/pilot_cssn2/hmc_marginal_failure_fold0_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sleepcp.models import build_model  # noqa: E402


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--shrink", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.cache) as d:
        data = {k: d[k] for k in d.files}
    state = torch.load(args.run, map_location="cpu", weights_only=True)
    norm = "batch" if any(k.endswith("num_batches_tracked") for k in state) else "group"
    model = build_model(args.model, norm=norm)
    model.load_state_dict(state)
    model.eval()
    lin = model.fc if args.model == "eegnet" else model.head[3]
    W = lin.weight.detach().numpy()
    b = lin.bias.detach().numpy()

    z_tr, y_tr = data["z_train"], data["y_train"]
    z_te, y_te, r_te = data["z_test"], data["y_test"], data["r_test"]
    cohort = np.bincount(y_tr, minlength=5) / y_tr.size
    mu_global = z_tr.mean(axis=0)
    mu_k = {k: z_tr[y_tr == k].mean(axis=0) for k in range(5)
            if (y_tr == k).sum() > 0}

    rows = []
    for name in sorted(set(r_te.tolist())):
        m = r_te == name
        z, y = z_te[m], y_te[m]
        p_bn = softmax(z @ W.T + b)
        p_mg = softmax((z - (z.mean(axis=0) - mu_global)) @ W.T + b)
        corr = np.zeros_like(z)
        for k, mk in mu_k.items():
            w = p_bn[:, k]
            total = float(w.sum())
            if total <= 1e-9:
                continue
            delta_k = (w @ z + args.shrink * mk) / (total + args.shrink) - mk
            corr += p_bn[:, [k]] * delta_k[None, :]
        p_cd = softmax((z - corr) @ W.T + b)
        rec_dist = np.bincount(y, minlength=5) / y.size
        acc_bn = float((p_bn.argmax(1) == y).mean())
        acc_mg = float((p_mg.argmax(1) == y).mean())
        acc_cd = float((p_cd.argmax(1) == y).mean())
        rows.append({
            "recording": str(name),
            "n": int(m.sum()),
            "js_from_cohort": js_divergence(rec_dist, cohort),
            "acc_bn": acc_bn,
            "acc_marginal": acc_mg,
            "acc_conditional": acc_cd,
            "delta_marginal": acc_mg - acc_bn,
            "delta_conditional": acc_cd - acc_bn,
            "stage_dist": rec_dist.tolist(),
        })

    js = np.array([r["js_from_cohort"] for r in rows])
    d_mg = np.array([r["delta_marginal"] for r in rows])
    d_cd = np.array([r["delta_conditional"] for r in rows])
    rho_mg, p_mg = spearmanr(js, d_mg)
    rho_cd, p_cd = spearmanr(js, d_cd)
    summary = {
        "n_recordings": len(rows),
        "mean_acc_bn": float(np.mean([r["acc_bn"] for r in rows])),
        "mean_acc_marginal": float(np.mean([r["acc_marginal"] for r in rows])),
        "mean_acc_conditional": float(np.mean([r["acc_conditional"] for r in rows])),
        "mean_delta_marginal": float(d_mg.mean()),
        "mean_delta_conditional": float(d_cd.mean()),
        "spearman_js_vs_delta_marginal": float(rho_mg),
        "spearman_p_marginal": float(p_mg),
        "spearman_js_vs_delta_conditional": float(rho_cd),
        "spearman_p_conditional": float(p_cd),
        "rows": rows,
    }
    print(f"recordings={len(rows)} "
          f"mean delta marginal={d_mg.mean():+.4f} conditional={d_cd.mean():+.4f}")
    print(f"spearman(js, delta) marginal={rho_mg:+.3f} (p={p_mg:.3g}); "
          f"conditional={rho_cd:+.3f} (p={p_cd:.3g})")
    for r in rows:
        print(f"{r['recording']:10s} n={r['n']:5d} js={r['js_from_cohort']:.4f} "
              f"bn={r['acc_bn']:.3f} marg={r['acc_marginal']:.3f} "
              f"cond={r['acc_conditional']:.3f} "
              f"dm={r['delta_marginal']:+.3f} dc={r['delta_conditional']:+.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
