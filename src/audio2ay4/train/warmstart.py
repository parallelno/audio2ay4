"""Supervised warm-start training loop (Plan A phase A2).

Pretrains the reverse player on ``(rendered audio → features, parsed-target AYState)`` pairs from
the shared data pipeline so it starts the reward phase already competent (design A.5). Building the
pairs renders the corpus through the emulator (heavy / native) — run that on the training machine.

The pure-tensor pieces (``collate``, ``train_step``) carry no emulator dependency, so the loop can
be unit-tested on CPU with synthetic batches (the §A.11 "overfit one batch" canary).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..config import RunConfig, TrainConfig
from ..data.pairing import TrainingPair, build_pair
from ..models.policy.network import ReversePlayer
from ..models.policy.spec import N_ENV_SHAPES, N_VOICES
from .targets import build_targets
from .warmstart_loss import WarmstartWeights, warmstart_loss

Sample = tuple[np.ndarray, dict[str, np.ndarray]]
Batch = tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]

_VOICE_KEYS = ("pitch", "volume", "tone", "noise", "env_use")
_FRAME_KEYS = ("noise_pitch", "env_rate", "env_retrig")


def pair_to_sample(pair: TrainingPair) -> Sample:
    """One training pair → ``(feats (T, dim), targets)`` for collation."""
    feats = np.asarray(pair.feats.feats, dtype=np.float32)
    targets = build_targets(pair.target_regs, pair.master_clock_hz, pair.frame_rate_hz)
    n = min(feats.shape[0], targets["env_shape"].shape[0])
    feats = feats[:n]
    targets = {k: v[..., :n] if v.ndim == 1 else v[:, :n] for k, v in targets.items()}
    return feats, targets


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


def train_warmstart(
    train_cfg: TrainConfig,
    ym_paths: list[str],
    *,
    device: str | None = None,
    log_every: int = 25,
) -> str:
    """Render → pair → pretrain the reverse player; save a checkpoint. Returns its path.

    Heavy: ``build_pair`` renders each YM through the emulator (runs on the training machine).
    """
    if not ym_paths:
        raise ValueError("train_warmstart needs at least one YM path")
    device = device or ("cuda" if (train_cfg.run.use_gpu and torch.cuda.is_available()) else "cpu")
    run: RunConfig = train_cfg.run

    import sys, time as _time
    total = len(ym_paths)
    print(f"Rendering {total} YM files …", flush=True)
    samples = []
    _t0 = _time.monotonic()
    for i, p in enumerate(ym_paths, 1):
        try:
            samples.append(pair_to_sample(build_pair(p, run, cache_dir=train_cfg.cache_dir)))
        except Exception as exc:
            print(f"[skip] {p}: {exc}", file=sys.stderr, flush=True)
        if i % 100 == 0 or i == total:
            elapsed = _time.monotonic() - _t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"  rendered {i}/{total} ({100*i//total}%) "
                f"| ok {len(samples)} | {rate:.1f} f/s | ETA {eta:.0f}s",
                flush=True,
            )
    if not samples:
        raise ValueError("No usable YM files after rendering (all failed)")
    dim = samples[0][0].shape[1]

    torch.manual_seed(run.seed)
    rng = np.random.default_rng(run.seed)
    hidden = int(run.extra.get("hidden", 128))
    net = ReversePlayer(in_dim=dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=train_cfg.lr)
    weights = WarmstartWeights()

    bs = max(1, min(train_cfg.batch_size, len(samples)))
    for step in range(1, train_cfg.max_steps + 1):
        idx = rng.choice(len(samples), size=bs, replace=len(samples) < bs)
        batch = collate([samples[i] for i in idx], device=device)
        total, parts = train_step(net, opt, batch, weights)
        if step % log_every == 0 or step == 1:
            print(f"step {step:>6}/{train_cfg.max_steps}  loss={total:.4f}  "
                  + " ".join(f"{k}={v:.3f}" for k, v in parts.items()))

    out = _checkpoint_path(train_cfg)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "in_dim": dim, "hidden": hidden}, out)
    print(f"Saved warm-start checkpoint → {out}")
    return out
