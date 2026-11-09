"""Prototype of stage-conditional batch normalisation (SCBN).

AdaBN replaces the running statistics of every normalisation layer with the
statistics of the whole recording.  SCBN keeps the stage distinction at every
layer: the mean and variance of each channel are estimated separately for wake,
N1, N2, N3 and REM, with the soft stage posterior of the unadapted model as the
only source of supervision, and each epoch is normalised with its own
posterior-weighted statistics.

Three passes over one recording:

1. the unadapted model produces a posterior for every epoch;
2. every normalisation layer accumulates soft stage-conditional first and
   second moments of its input;
3. each epoch is normalised with the mixture of those moments given its own
   posterior.

This is the layer-wise counterpart of SCSN.  It is closed-form, uses no label
and changes no weight; the only choice is the shrinkage pseudo-count with
which a stage's moments are pulled toward the cohort's running statistics.

Example
-------
python scripts/scbn_probe.py --model eegnet --fold 0 --seed 0 \
    --data-dir data/processed/sleep-edf-20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adabn_final as af  # noqa: E402

from sleepcp.data import load_dataset  # noqa: E402
from sleepcp.models import build_model  # noqa: E402

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


def bn_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in model.modules() if isinstance(m, BN_TYPES)]


def channel_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample, per-channel mean and second moment of a BN input."""
    if x.dim() == 2:                      # (B, C) or (B, L)
        if x.size(1) == 1:
            return x[:, 0], x[:, 0] ** 2
        return x, x ** 2
    dims = [2] if x.dim() == 3 else [2, 3]
    return x.mean(dim=dims), (x ** 2).mean(dim=dims)


def posteriors(model, model_name, x, device, batch_size=256):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.as_tensor(x[start : start + batch_size])[:, None, :].to(device)
            logits = model(batch)
            out.append(af.softmax(logits.cpu().numpy()))
    return np.concatenate(out)


def stage_stats(model, x, p, device, batch_size=256):
    """Soft stage-conditional moments of every BN layer's input."""
    mods = bn_modules(model)
    sums = [torch.zeros(5, m.num_features, dtype=torch.float64) for m in mods]
    sqs = [torch.zeros(5, m.num_features, dtype=torch.float64) for m in mods]
    mass = [torch.zeros(5, dtype=torch.float64) for _ in mods]
    handles = []
    current = {}

    def make_hook(idx):
        def hook(module, inputs):
            xin = inputs[0].detach()
            mean, sq = channel_moments(xin)
            mean = mean.double().cpu()
            sq = sq.double().cpu()
            w = current["p"].double().cpu()
            sums[idx] += w.T @ mean
            sqs[idx] += w.T @ sq
            mass[idx] += w.sum(dim=0)
        return hook

    for idx, mod in enumerate(mods):
        handles.append(mod.register_forward_pre_hook(make_hook(idx)))
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            current["p"] = torch.as_tensor(p[start:stop])
            batch = torch.as_tensor(x[start:stop])[:, None, :].to(device)
            model(batch)
    for handle in handles:
        handle.remove()
    return mods, sums, sqs, mass


def scbn_stats(mods, sums, sqs, mass, shrink):
    """Shrunk per-stage mean and variance for each layer."""
    stats = []
    for mod, s, q, w in zip(mods, sums, sqs, mass):
        m0 = mod.running_mean.double().cpu()
        v0 = mod.running_var.double().cpu()
        w = w.clamp_min(0.0)
        mu = (s + shrink * m0[None, :]) / (w[:, None] + shrink)
        second = (q + shrink * (v0 + m0 ** 2)[None, :]) / (w[:, None] + shrink)
        var = (second - mu ** 2).clamp_min(1e-6)
        stats.append((mu, var))
    return stats


class SCBN:
    """Temporarily replace each BN layer's forward with a per-epoch mixture."""

    def __init__(self, mods):
        self.mods = mods
        self.original = [m.forward for m in mods]
        self.mu = [None] * len(mods)
        self.var = [None] * len(mods)

    def __enter__(self):
        for idx, mod in enumerate(self.mods):
            mod.forward = self._make_forward(idx, mod)
        return self

    def __exit__(self, *exc):
        for mod, fwd in zip(self.mods, self.original):
            mod.forward = fwd
        return False

    def _make_forward(self, idx, module):
        def forward(x):
            mu, var = self.mu[idx], self.var[idx]
            if mu is None:
                return self.original[idx](x)
            mu = mu.to(x.device).to(x.dtype)
            var = var.to(x.device).to(x.dtype)
            shape = [1] * x.dim()
            shape[1] = x.size(1)
            mean = mu.view(-1, x.size(1), *([1] * (x.dim() - 2)))
            variance = var.view_as(mean)
            y = (x - mean) / torch.sqrt(variance + module.eps)
            if module.affine:
                y = y * module.weight.view(shape) + module.bias.view(shape)
            return y
        return forward

    def set_batch(self, stats, indices):
        for idx, (mu, var) in enumerate(stats):
            self.mu[idx] = mu[indices]
            self.var[idx] = var[indices]


def cohort_stage_means(model, recordings, names, device, batch_size=256):
    """Per-layer cohort stage means of the pre-normalisation activations."""
    mods = bn_modules(model)
    sums = [torch.zeros(5, m.num_features, dtype=torch.float64) for m in mods]
    mass = [torch.zeros(5, dtype=torch.float64) for _ in mods]
    handles = []
    current = {}

    def make_hook(idx):
        def hook(module, inputs):
            mean, _ = channel_moments(inputs[0].detach())
            mean = mean.double().cpu()
            w = torch.nn.functional.one_hot(current["y"], num_classes=5).double()
            sums[idx] += w.T @ mean
            mass[idx] += w.sum(dim=0)
        return hook

    for idx, mod in enumerate(mods):
        handles.append(mod.register_forward_pre_hook(make_hook(idx)))
    model.eval()
    with torch.no_grad():
        for name in names:
            rec = recordings[name]
            keep = rec.y >= 0
            x, y = rec.x[keep], rec.y[keep]
            for start in range(0, x.shape[0], batch_size):
                stop = min(start + batch_size, x.shape[0])
                current["y"] = torch.as_tensor(y[start:stop])
                batch = torch.as_tensor(x[start:stop])[:, None, :].to(device)
                model(batch)
    for handle in handles:
        handle.remove()
    return [(s / w.clamp_min(1.0)[:, None]) for s, w in zip(sums, mass)]


def recording_moments(model, x, p, device, batch_size=256):
    """Recording-wide and soft stage-conditional moments of BN inputs."""
    mods = bn_modules(model)
    gsum = [torch.zeros(m.num_features, dtype=torch.float64) for m in mods]
    gsq = [torch.zeros(m.num_features, dtype=torch.float64) for m in mods]
    ssum = [torch.zeros(5, m.num_features, dtype=torch.float64) for m in mods]
    mass = [torch.zeros(5, dtype=torch.float64) for _ in mods]
    count = 0
    handles = []
    current = {}

    def make_hook(idx):
        def hook(module, inputs):
            mean, sq = channel_moments(inputs[0].detach())
            mean = mean.double().cpu()
            sq = sq.double().cpu()
            gsum[idx] += mean.sum(dim=0)
            gsq[idx] += sq.sum(dim=0)
            w = current["p"].double().cpu()
            ssum[idx] += w.T @ mean
            mass[idx] += w.sum(dim=0)
        return hook

    for idx, mod in enumerate(mods):
        handles.append(mod.register_forward_pre_hook(make_hook(idx)))
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            current["p"] = torch.as_tensor(p[start:stop])
            batch = torch.as_tensor(x[start:stop])[:, None, :].to(device)
            model(batch)
            count += stop - start
    for handle in handles:
        handle.remove()
    gmean = [s / count for s in gsum]
    gvar = [q / count - (s / count) ** 2 for q, s in zip(gsq, gsum)]
    stage_mean = [s / w.clamp_min(1e-6)[:, None] for s, w in zip(ssum, mass)]
    return gmean, gvar, stage_mean, mass


def per_epoch_stats(stats, p):
    """Posterior-weighted mixture of stage moments for each epoch."""
    out = []
    for mu, var in stats:
        mu_i = torch.as_tensor(p, dtype=torch.float64) @ mu      # (n, C)
        var_i = torch.as_tensor(p, dtype=torch.float64) @ var
        out.append((mu_i, var_i))
    return out


def run_scbn(model, model_name, x, p, device, shrink, batch_size=256):
    mods, sums, sqs, mass = stage_stats(model, x, p, device, batch_size)
    stats = scbn_stats(mods, sums, sqs, mass, shrink)
    per_epoch = per_epoch_stats(stats, p)
    feats = []
    model.eval()
    with SCBN(mods) as scbn:
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                stop = min(start + batch_size, x.shape[0])
                scbn.set_batch(per_epoch, torch.arange(start, stop))
                batch = torch.as_tensor(x[start:stop])[:, None, :].to(device)
                if model_name == "eegnet":
                    z = model.bn1(model.conv1(batch[:, :, None, :]))
                    z = model.bn2(model.conv2(z))
                    z = torch.nn.functional.elu(z)
                    z = model.drop1(model.pool1(z))
                    z = model.conv4(model.conv3(z))
                    z = model.bn3(z)
                    z = torch.nn.functional.elu(z)
                    z = model.drop2(model.pool2(z))
                else:
                    z = model.features(batch)
                    z = model.head[0](z)
                    z = model.head[1](z)
                    z = model.head[2](z)
                z = torch.flatten(z, 1)
                feats.append(z.cpu().numpy())
    return np.concatenate(feats)


def run_scbn_shift(model, model_name, x, p, cohort_means, device,
                   batch_size=256, clip=None):
    """AdaBN at every layer, with the stage-conditional offset removed."""
    gmean, gvar, stage_mean, mass = recording_moments(model, x, p, device, batch_size)
    per_epoch = []
    for gm, gv, sm, cm in zip(gmean, gvar, stage_mean, cohort_means):
        shift = torch.as_tensor(p, dtype=torch.float64) @ (sm - cm)
        if clip is not None:
            norm = shift.norm(dim=1, keepdim=True)
            shift = shift * (clip / norm.clamp_min(clip))
        # SCSN at the layer: subtract only the stage-conditional part of the
        # shift, relative to the cohort prototype.  The recording-wide offset
        # is already carried by the normalisation itself.
        mu = shift
        var = gv[None, :].expand_as(mu)
        per_epoch.append((mu, var))
    mods = bn_modules(model)
    feats = []
    model.eval()
    with SCBN(mods) as scbn:
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                stop = min(start + batch_size, x.shape[0])
                scbn.set_batch(per_epoch, torch.arange(start, stop))
                batch = torch.as_tensor(x[start:stop])[:, None, :].to(device)
                if model_name == "eegnet":
                    z = model.bn1(model.conv1(batch[:, :, None, :]))
                    z = model.bn2(model.conv2(z))
                    z = torch.nn.functional.elu(z)
                    z = model.drop1(model.pool1(z))
                    z = model.conv4(model.conv3(z))
                    z = model.bn3(z)
                    z = torch.nn.functional.elu(z)
                    z = model.drop2(model.pool2(z))
                else:
                    z = model.features(batch)
                    z = model.head[0](z)
                    z = model.head[1](z)
                    z = model.head[2](z)
                z = torch.flatten(z, 1)
                feats.append(z.cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shrink", type=float, default=10.0)
    parser.add_argument("--arms", default="bn,adabn,scbn,scbn2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.root / args.model / f"fold{args.fold}_seed{args.seed}"
    stem = f"{args.model}_fold{args.fold}_seed{args.seed}"
    run_json = json.loads((run_dir / f"{stem}.json").read_text(encoding="utf-8"))
    test_subjects = sorted(set(run_json["test_subjects"]))
    train_subjects = sorted(set(run_json["train_subjects"]) | set(run_json["val_subjects"]))

    state = torch.load(run_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    norm = "batch" if any(k.endswith("num_batches_tracked") for k in state) else "group"
    model = build_model(args.model, norm=norm).to(args.device)
    model.load_state_dict(state)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    W = af.final_linear(model, args.model).weight.detach().cpu().numpy()
    b = af.final_linear(model, args.model).bias.detach().cpu().numpy()

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    names = sorted({r.recording for r in recordings.values()
                    if r.subject in set(test_subjects)})
    train_names = sorted({r.recording for r in recordings.values()
                          if r.subject in set(train_subjects)})
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    cohort_means = None
    if any(a.startswith("scbshift") for a in arms):
        cohort_means = cohort_stage_means(model, recordings, train_names,
                                          args.device)
        model.load_state_dict(initial)
    results = {a: [] for a in arms}
    labels = []
    for name in names:
        rec = recordings[name]
        keep = rec.y >= 0
        x = rec.x[keep]
        y = rec.y[keep]
        labels.append(y)
        p = posteriors(model, args.model, x, args.device)
        for arm in arms:
            model.load_state_dict(initial)
            if arm == "bn":
                z, _ = af.penultimate(model, args.model, x, args.device)
            elif arm == "adabn":
                af.adapt_bn(model, initial, x, args.device)
                z, _ = af.penultimate(model, args.model, x, args.device)
            elif arm in ("scbn", "scbn2"):
                p_use = p
                z = run_scbn(model, args.model, x, p_use, args.device, args.shrink)
                if arm == "scbn2":
                    model.load_state_dict(initial)
                    p2 = af.softmax(z @ W.T + b)
                    z = run_scbn(model, args.model, x, p2, args.device, args.shrink)
            elif arm == "scbshift":
                z = run_scbn_shift(model, args.model, x, p, cohort_means,
                                   args.device)
            elif arm == "scbshift_iter":
                z = run_scbn_shift(model, args.model, x, p, cohort_means,
                                   args.device)
                p2 = af.softmax(z @ W.T + b)
                model.load_state_dict(initial)
                z = run_scbn_shift(model, args.model, x, p2, cohort_means,
                                   args.device)
            else:
                raise ValueError(arm)
            results[arm].append(z)
        model.load_state_dict(initial)
        print(f"{name}: n={keep.sum()} done", flush=True)

    labels = np.concatenate(labels)
    for arm in arms:
        z = np.concatenate(results[arm])
        pred = (z @ W.T + b).argmax(axis=1)
        print(f"{arm:8s} acc={float((pred == labels).mean()):.4f}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, labels=labels,
                            **{a: np.concatenate(results[a]) for a in arms})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
