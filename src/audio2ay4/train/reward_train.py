"""Regime-1 reward training loop (design A.3 / A.4): differentiable analysis-by-synthesis.

Fine-tunes a (warm-started) reverse player so that **re-rendering its output sounds like the input**.
Per step: features → ``E`` → relaxed continuous controls (``models.policy.relax``) → differentiable
emulator (``chip.diff``) → multi-scale spectral reconstruction loss against the target audio. The
target is the corpus performance rendered through the *same* differentiable emulator from its
ground-truth registers, so a perfect policy drives the loss to ~0 (the §A.11 canary floor) and the
gradient is about the controls, not the trusted-vs-twin emulator gap (that is A5's real-audio job).

``E`` is normally initialised from an A2 warm-start checkpoint (``--init``); training only moves the
weights, never the legality contract (the deterministic compiler still owns the final registers).

The corpus loading mirrors ``warmstart``: ``build_pair`` (torch-free, cached) fans out across
processes; the torch training step runs on the main process / GPU.
"""

from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from .. import features
from ..chip.diff import DiffAyEmulator, unpack_regs
from ..config import RunConfig, TrainConfig
from ..data.augment import augment_audio
from ..data.pairing import build_pair
from ..models.policy.network import ReversePlayer
from ..models.policy.relax import controls_from_heads
from ..repr.state import AudioBuffer
from .reward import RewardWeights, jitter_penalty, multiscale_stft_loss

# One reward example: features (T, dim), ground-truth registers (T, 16), and the chip clock/rate.
RewardSample = tuple[np.ndarray, np.ndarray, int, int]


def _load_one(task: tuple[str, RunConfig, str | None]) -> RewardSample:
    """Process-pool worker: render (or load-from-cache) one YM → reward sample (torch-free)."""
    path, run, cache_dir = task
    pair = build_pair(path, run, cache_dir=cache_dir)
    feats = np.asarray(pair.feats.feats, dtype=np.float32)
    regs = np.asarray(pair.target_regs, dtype=np.uint8)
    n = min(feats.shape[0], regs.shape[0])
    return feats[:n], regs[:n], int(pair.master_clock_hz), int(pair.frame_rate_hz)


def load_reward_samples(
    ym_paths: list[str],
    run: RunConfig,
    cache_dir: str | None,
    *,
    workers: int | None = None,
    log_every: int = 100,
) -> list[RewardSample]:
    """Render every YM to a reward sample (parallel, cached); skip unparseable files."""
    total = len(ym_paths)
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    n_workers = max(1, min(workers, total))
    print(f"Loading {total} reward samples (workers={n_workers}, cache={cache_dir or 'off'}) …",
          flush=True)

    results: dict[int, RewardSample] = {}
    t0 = time.monotonic()
    done = 0

    def _tick() -> None:
        elapsed = time.monotonic() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"  loaded {done}/{total} ({100 * done // total}%) | ok {len(results)} "
              f"| {rate:.1f} f/s | ETA {eta:.0f}s", flush=True)

    if n_workers <= 1:
        for i, path in enumerate(ym_paths):
            try:
                results[i] = _load_one((path, run, cache_dir))
            except Exception as exc:  # noqa: BLE001 — one bad tune must not kill the run
                print(f"[skip] {path}: {exc}", file=sys.stderr, flush=True)
            done += 1
            if done % log_every == 0 or done == total:
                _tick()
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_load_one, (p, run, cache_dir)): i
                       for i, p in enumerate(ym_paths)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # noqa: BLE001 — skip & log, keep going
                    print(f"[skip] {ym_paths[i]}: {exc}", file=sys.stderr, flush=True)
                done += 1
                if done % log_every == 0 or done == total:
                    _tick()

    print(f"Load done: {len(results)} ok, {total - len(results)} skipped.", flush=True)
    return [results[i] for i in sorted(results)]


def _crop(sample: RewardSample, window: int | None, rng: np.random.Generator) -> RewardSample:
    """Random fixed-length time window over features + registers (full sample if shorter)."""
    feats, regs, mclk, fr = sample
    t = feats.shape[0]
    if window is None or t <= window:
        return sample
    s = int(rng.integers(0, t - window + 1))
    return feats[s:s + window], regs[s:s + window], mclk, fr


def _lr_at(step: int, base_lr: float, max_steps: int, warmup: int) -> float:
    """Linear warmup then cosine decay to ~0 (same schedule as the warm-start)."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _augmented_input(
    target_audio: torch.Tensor,
    run: RunConfig,
    rng: np.random.Generator,
    strength: float,
) -> np.ndarray:
    """Degrade the clean reference render to a SUNO-like clip and re-extract numpy log-mel.

    Mirrors the inference front-end exactly (same ``features.extract``), so the encoder trains on the
    feature distribution it will actually meet at deployment — just coloured to bridge the domain gap.
    """
    pcm = target_audio.detach().to("cpu").numpy()
    aug = augment_audio(pcm, run.sample_rate, rng, strength=strength)
    audio = AudioBuffer(pcm=aug, sample_rate=run.sample_rate, duration_s=len(aug) / float(run.sample_rate))
    return features.extract(audio, run).feats                            # (T, dim) float32


def reward_forward(
    net: ReversePlayer,
    emulator: DiffAyEmulator,
    samples: list[RewardSample],
    *,
    device: str,
    weights: RewardWeights,
    tau: float,
    run: RunConfig | None = None,
    augment: bool = False,
    aug_strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Analysis-by-synthesis forward pass over a (cropped) batch → ``(total_loss, parts)``.

    Each tune is rendered independently (clock/rate vary per tune); reconstructions are stacked to a
    common length for the batched spectral term, while the jitter term is averaged over the per-tune
    relaxed controls (both differentiable through the rendered audio).

    When ``augment`` is set (design A.5), the encoder input is not the cached clean features but a
    SUNO-style degraded re-render of the *same* reference audio (``_augmented_input``); the reward
    target stays the clean reference, so the encoder learns to recover the AY controls despite
    real-world coloration. Requires ``run`` and ``rng``.
    """
    net.train()
    recon_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []
    jitter = torch.zeros((), device=device)

    for feats, regs, mclk, fr in samples:
        with torch.no_grad():
            target = emulator.render(unpack_regs(regs).to(device), mclk, fr)
        if augment:
            assert run is not None and rng is not None, "augment mode needs run + rng"
            feats = _augmented_input(target, run, rng, aug_strength)
        x = torch.from_numpy(np.ascontiguousarray(feats.T)).unsqueeze(0).to(device)  # (1, dim, T)
        heads = net(x)
        controls = controls_from_heads(heads, float(mclk), tau=tau)
        recon = emulator.render(controls.select(0), mclk, fr)            # (N,) differentiable
        recon_list.append(recon)
        target_list.append(target)
        jitter = jitter + jitter_penalty(controls)

    n = min(min(r.shape[0] for r in recon_list), min(t.shape[0] for t in target_list))
    recon_b = torch.stack([r[:n] for r in recon_list], dim=0)            # (B, n)
    target_b = torch.stack([t[:n] for t in target_list], dim=0)

    spec = multiscale_stft_loss(recon_b, target_b)
    jitter = jitter / max(1, len(samples))
    total = weights.spectral * spec + weights.jitter * jitter
    parts = {"spectral": float(spec.detach()), "jitter": float(jitter.detach())}
    return total, parts


def _checkpoint_path(train_cfg: TrainConfig) -> str:
    return train_cfg.run.extra.get("checkpoint") or os.path.join(
        train_cfg.cache_dir or ".cache", "reward_rl.pt"
    )


def train_reward(
    train_cfg: TrainConfig,
    ym_paths: list[str],
    *,
    init_checkpoint: str | None = None,
    device: str | None = None,
    workers: int | None = None,
    window: int | None = 256,
    weights: RewardWeights | None = None,
    tau: float = 1.0,
    augment: bool = False,
    aug_strength: float = 1.0,
    max_partials: int = 24,
    oversample: int = 2,
    log_every: int = 25,
) -> str:
    """Load corpus → Regime-1 reward-train the reverse player → save a checkpoint. Returns its path."""
    if not ym_paths:
        raise ValueError("train_reward needs at least one YM path")
    device = device or ("cuda" if (train_cfg.run.use_gpu and torch.cuda.is_available()) else "cpu")
    run: RunConfig = train_cfg.run
    weights = weights or RewardWeights()

    samples = load_reward_samples(ym_paths, run, train_cfg.cache_dir, workers=workers)
    if not samples:
        raise ValueError("No usable YM files after loading (all failed)")
    dim = samples[0][0].shape[1]

    torch.manual_seed(run.seed)
    rng = np.random.default_rng(run.seed)
    hidden = int(run.extra.get("hidden", 128))
    net = ReversePlayer(in_dim=dim, hidden=hidden).to(device)
    if init_checkpoint:
        state = torch.load(init_checkpoint, map_location="cpu")
        net.load_state_dict(state.get("model", state))
        print(f"Initialised from warm-start checkpoint {init_checkpoint}")
    opt = torch.optim.AdamW(net.parameters(), lr=train_cfg.lr, weight_decay=0.0)

    emulator = DiffAyEmulator(
        render_sr=run.sample_rate, oversample=oversample, max_partials=max_partials
    ).to(device)

    bs = max(1, min(train_cfg.batch_size, len(samples)))
    base_lr = train_cfg.lr
    warmup = max(1, min(200, train_cfg.max_steps // 10))
    print(
        f"Reward-training on {device}: {len(samples)} tunes | batch {bs} "
        f"| window {window or 'full'} | {train_cfg.max_steps} steps "
        f"| lr {base_lr:.1e} (warmup {warmup}, cosine) "
        f"| w_spec {weights.spectral:.2f} w_jit {weights.jitter:.3f} tau {tau:.2f} "
        f"| augment {'on s=' + format(aug_strength, '.2f') if augment else 'off'} "
        f"| emu sr {run.sample_rate} x{oversample}, partials {max_partials}",
        flush=True,
    )

    ema: float | None = None
    t0 = time.monotonic()
    for step in range(1, train_cfg.max_steps + 1):
        lr = _lr_at(step, base_lr, train_cfg.max_steps, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.choice(len(samples), size=bs, replace=len(samples) < bs)
        batch = [_crop(samples[i], window, rng) for i in idx]
        total, parts = reward_forward(net, emulator, batch, device=device,
                                      weights=weights, tau=tau, run=run,
                                      augment=augment, aug_strength=aug_strength, rng=rng)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        loss = float(total.detach())
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if step <= 3 or step % log_every == 0 or step == train_cfg.max_steps:
            elapsed = time.monotonic() - t0
            sps = step / elapsed if elapsed > 0 else 0.0
            eta = (train_cfg.max_steps - step) / sps if sps > 0 else 0.0
            print(
                f"step {step:>6}/{train_cfg.max_steps}  loss={loss:8.4f} avg={ema:8.4f}  "
                f"| {sps:4.1f} it/s | ETA {eta:4.0f}s | lr {lr:.1e}  "
                + " ".join(f"{k}={v:.4f}" for k, v in parts.items()),
                flush=True,
            )

    out = _checkpoint_path(train_cfg)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "in_dim": dim, "hidden": hidden}, out)
    print(f"Saved reward checkpoint → {out}")
    return out


__all__ = [
    "RewardSample",
    "load_reward_samples",
    "reward_forward",
    "train_reward",
]
