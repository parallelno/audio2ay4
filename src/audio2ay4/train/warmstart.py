"""Supervised warm-start training loop (Plan A phase A2).

Pretrains the reverse player on ``(rendered audio → features, parsed-target AYState)`` pairs from
the shared data pipeline so it starts the reward phase already competent (design A.5). Building the
pairs renders the corpus through the emulator (heavy / native) — run that on the training machine.

The pure-tensor pieces (``collate``, ``train_step``) carry no emulator dependency, so the loop can
be unit-tested on CPU with synthetic batches (the §A.11 "overfit one batch" canary).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from ..config import RunConfig, TrainConfig
from ..models.policy.network import ReversePlayer
from ..models.policy.spec import N_ENV_SHAPES, N_VOICES
from .render import Sample, pair_to_sample, render_samples
from .warmstart_loss import WarmstartWeights, warmstart_loss

Batch = tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]

_VOICE_KEYS = ("pitch", "volume", "tone", "noise", "env_use")
_FRAME_KEYS = ("noise_pitch", "env_rate", "env_retrig")

# ``pair_to_sample`` is imported from .render and re-exported here for existing call sites/tests.
__all__ = ["Sample", "pair_to_sample", "render_samples", "collate", "train_step", "train_warmstart"]


def collate(samples: list[Sample], device: str = "cpu") -> Batch:
    """Pad a list of samples to the batch's max length; build the per-frame ``pad_mask``."""
    batch = len(samples)
    dim = samples[0][0].shape[1]
    max_t = max(s[0].shape[0] for s in samples)

    x = np.zeros((batch, dim, max_t), np.float32)
    pad = np.zeros((batch, max_t), np.float32)
    voice = {k: np.zeros((batch, N_VOICES, max_t), np.float32) for k in _VOICE_KEYS}
    frame = {k: np.zeros((batch, max_t), np.float32) for k in _FRAME_KEYS}
    env_shape = np.zeros((batch, max_t), np.int64)
    # env_rate must be positive everywhere (loss takes log); fill padding with 1.0 Hz.
    frame["env_rate"][:] = 1.0

    for b, (feats, tgt) in enumerate(samples):
        t = feats.shape[0]
        x[b, :, :t] = feats.T
        pad[b, :t] = 1.0
        for k in _VOICE_KEYS:
            voice[k][b, :, :t] = tgt[k]
        for k in _FRAME_KEYS:
            frame[k][b, :t] = tgt[k]
        env_shape[b, :t] = tgt["env_shape"]

    targets: dict[str, torch.Tensor] = {}
    for k in _VOICE_KEYS:
        targets[k] = torch.from_numpy(voice[k]).to(device)
    for k in _FRAME_KEYS:
        targets[k] = torch.from_numpy(frame[k]).to(device)
    targets["env_shape"] = torch.from_numpy(env_shape).to(device)
    assert targets["env_shape"].max().item() < N_ENV_SHAPES
    return torch.from_numpy(x).to(device), targets, torch.from_numpy(pad).to(device)


def train_step(
    net: ReversePlayer,
    opt: torch.optim.Optimizer,
    batch: Batch,
    weights: WarmstartWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """One optimisation step on a padded batch. Returns ``(total_loss, parts)``."""
    net.train()
    x, targets, pad_mask = batch
    heads = net(x)
    total, parts = warmstart_loss(heads, targets, pad_mask, weights)
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()
    return float(total.detach()), parts


def _checkpoint_path(train_cfg: TrainConfig) -> str:
    return train_cfg.run.extra.get("checkpoint") or os.path.join(
        train_cfg.cache_dir or ".cache", "warmstart_rl.pt"
    )


def _crop(sample: Sample, window: int | None, rng: np.random.Generator) -> Sample:
    """Random fixed-length time window of a sample (full sample if shorter than ``window``).

    Training on whole songs makes each step pad to the batch's longest tune (thousands of frames),
    which is both slow and mostly wasted compute. Cropping bounds per-step cost and keeps batches
    uniform, so throughput no longer depends on the longest tune in the draw.
    """
    feats, targets = sample
    t = feats.shape[0]
    if window is None or t <= window:
        return sample
    s = int(rng.integers(0, t - window + 1))
    e = s + window
    cropped = {k: (v[..., s:e] if v.ndim == 1 else v[:, s:e]) for k, v in targets.items()}
    return feats[s:e], cropped


def _lr_at(step: int, base_lr: float, max_steps: int, warmup: int) -> float:
    """Linear warmup for ``warmup`` steps, then cosine decay to ~0 by ``max_steps``."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def _evaluate(
    net: ReversePlayer,
    val_samples: list[Sample],
    *,
    window: int | None,
    bs: int,
    weights: WarmstartWeights,
    device: str,
    seed: int,
    n_batches: int = 8,
) -> float:
    """Mean warm-start loss over a few fixed validation batches (eval mode, no grad)."""
    net.eval()
    vrng = np.random.default_rng(seed + 1)
    losses = []
    for _ in range(n_batches):
        k = min(bs, len(val_samples))
        idx = vrng.choice(len(val_samples), size=k, replace=len(val_samples) < k)
        x, targets, pad = collate([_crop(val_samples[i], window, vrng) for i in idx], device=device)
        total, _ = warmstart_loss(net(x), targets, pad, weights)
        losses.append(float(total))
    net.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_warmstart(
    train_cfg: TrainConfig,
    ym_paths: list[str],
    *,
    device: str | None = None,
    workers: int | None = None,
    window: int | None = 512,
    val_frac: float = 0.05,
    val_every: int = 250,
    log_every: int = 25,
) -> str:
    """Render → pair → pretrain the reverse player; save a checkpoint. Returns its path.

    Heavy: ``build_pair`` renders each YM through the emulator (runs on the training machine).
    Rendering is parallelised across ``workers`` processes (``None``/0 = all CPUs).
    """
    if not ym_paths:
        raise ValueError("train_warmstart needs at least one YM path")
    device = device or ("cuda" if (train_cfg.run.use_gpu and torch.cuda.is_available()) else "cpu")
    run: RunConfig = train_cfg.run

    samples = render_samples(ym_paths, run, train_cfg.cache_dir, workers=workers)
    if not samples:
        raise ValueError("No usable YM files after rendering (all failed)")
    dim = samples[0][0].shape[1]

    torch.manual_seed(run.seed)
    rng = np.random.default_rng(run.seed)
    hidden = int(run.extra.get("hidden", 128))
    net = ReversePlayer(in_dim=dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=train_cfg.lr)
    weights = WarmstartWeights()

    # Hold out whole tunes for an honest convergence signal (per-step train loss is very noisy).
    n_val = min(int(len(samples) * val_frac) if val_frac > 0 else 0, len(samples) - 1)
    train_samples = samples[: len(samples) - n_val] if n_val > 0 else samples
    val_samples = samples[len(samples) - n_val :] if n_val > 0 else []

    bs = max(1, min(train_cfg.batch_size, len(train_samples)))
    base_lr = train_cfg.lr
    warmup = max(1, min(500, train_cfg.max_steps // 10))
    print(
        f"Training on {device}: {len(train_samples)} train / {len(val_samples)} val tunes "
        f"| batch {bs} | window {window or 'full'} | {train_cfg.max_steps} steps "
        f"| lr {base_lr:.1e} (warmup {warmup}, cosine)",
        flush=True,
    )

    ema: float | None = None
    t0 = time.monotonic()
    for step in range(1, train_cfg.max_steps + 1):
        lr = _lr_at(step, base_lr, train_cfg.max_steps, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.choice(len(train_samples), size=bs, replace=len(train_samples) < bs)
        batch = collate([_crop(train_samples[i], window, rng) for i in idx], device=device)
        total, parts = train_step(net, opt, batch, weights)
        ema = total if ema is None else 0.98 * ema + 0.02 * total
        if step <= 3 or step % log_every == 0 or step == train_cfg.max_steps:
            elapsed = time.monotonic() - t0
            sps = step / elapsed if elapsed > 0 else 0.0
            eta = (train_cfg.max_steps - step) / sps if sps > 0 else 0.0
            msg = (
                f"step {step:>6}/{train_cfg.max_steps}  loss={total:8.2f} avg={ema:8.2f}  "
                f"| {sps:4.1f} it/s | ETA {eta:4.0f}s | lr {lr:.1e}  "
                + " ".join(f"{k}={v:.3f}" for k, v in parts.items())
            )
            if val_samples and (step % val_every == 0 or step == train_cfg.max_steps):
                vloss = _evaluate(net, val_samples, window=window, bs=bs,
                                  weights=weights, device=device, seed=run.seed)
                msg = f"{msg}  || val={vloss:.2f}"
            print(msg, flush=True)

    out = _checkpoint_path(train_cfg)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "in_dim": dim, "hidden": hidden}, out)
    print(f"Saved warm-start checkpoint → {out}")
    return out
