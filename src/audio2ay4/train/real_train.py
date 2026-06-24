"""Regime A5 — *real-audio* analysis-by-synthesis (design A.5 / the §A roadmap "Tier 1" lever).

Unlike :mod:`reward_train` (which reconstructs a corpus YM through the *same* differentiable twin —
self-reconstruction, whose reward optima drift off the supervised mapping and regressed in v1/v3/v8),
this regime trains directly against **real audio**: SUNO-generated chiptune clips. There is **no YM
target**. Per step::

    real_audio → features → E → relaxed controls → diff-twin.render → recon
    loss = chroma_loss(recon, real_audio) + onset_loss(recon, real_audio)        (timbre-invariant)

The reward is purely perceptual agreement (pitch-class melody + onset rhythm) between the AY
re-render and the *real input the user actually fed in* — i.e. it optimises the exact quantity
``eval.metrics`` measures, on the exact distribution we deploy on. This removes the train/eval
distribution mismatch that capped every prior reward lever.

The spectral-magnitude term is **off by default** here: linear AY-square-wave magnitude vs. a rich
SUNO synth is dominated by timbre mismatch (perceptually invalid — the same reason ``spec_dist`` was
retired), so we lean on the timbre-invariant chroma/onset terms.

This is a *sizing harness* (Stage 0 overfit → Stage 1 go/no-go): clips are split into train / held-out
**by prompt stem** so SUNO regenerations of one prompt (``Pixel Drift (1..4)``) never straddle the
split. After training it scores the held-out clips with both the init checkpoint and the new one, so
the verdict is a clean apples-to-apples delta on unseen audio.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

from .. import features, io
from ..chip.diff import DiffAyEmulator
from ..config import RunConfig, TrainConfig
from ..models.policy.network import ReversePlayer
from ..models.policy.relax import controls_from_heads
from .reward import RewardWeights, chroma_loss, jitter_penalty, multiscale_stft_loss, onset_loss

# One real example: prompt-stem, features (T, dim), target PCM (M,), chip clock / frame rate.
RealSample = tuple[str, np.ndarray, np.ndarray, int, int]

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
_VARIANT_RE = re.compile(r"\s*\(\d+\)$")          # trailing " (1)", " (2)", … (SUNO regen suffix)


def find_audio_files(path: str) -> list[str]:
    """Every decodable audio file under ``path`` (a directory) or ``path`` itself (a single file)."""
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(_AUDIO_EXTS)
        )
    return [path] if path.lower().endswith(_AUDIO_EXTS) else []


def prompt_stem(path: str) -> str:
    """Normalise a filename to its prompt identity so SUNO regenerations group together.

    ``"Pixel Drift (3).mp3"`` / ``"Pixel_Drift.mp3"`` / ``"pixel drift.mp3"`` → ``"pixel drift"``.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace("_", " ")
    name = _VARIANT_RE.sub("", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def split_by_stem(
    paths: list[str], heldout_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split files into (train, held-out) **by prompt stem** — whole groups never straddle the split.

    Greedily assigns shuffled stem-groups to held-out until ``heldout_frac`` of *clips* is reached,
    so two regenerations of one prompt always land on the same side (no near-duplicate leakage).
    """
    groups: dict[str, list[str]] = {}
    for p in paths:
        groups.setdefault(prompt_stem(p), []).append(p)
    stems = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    total = len(paths)
    want = int(round(heldout_frac * total))
    heldout: list[str] = []
    train: list[str] = []
    for s in stems:
        if len(heldout) < want and len(train) >= 0:
            # keep at least one stem for training: don't let held-out swallow everything
            if len(heldout) + len(groups[s]) <= total - 1:
                heldout.extend(sorted(groups[s]))
                continue
        train.extend(sorted(groups[s]))
    return sorted(train), sorted(heldout)


def load_real_samples(
    paths: list[str], run: RunConfig, *, log_every: int = 10
) -> list[RealSample]:
    """Decode each clip → (stem, features, PCM, clock, frame-rate). Skips files that fail to decode."""
    out: list[RealSample] = []
    total = len(paths)
    print(f"Loading {total} real-audio samples (sr={run.sample_rate}) …", flush=True)
    for i, p in enumerate(paths, 1):
        try:
            audio = io.decode(p, target_sr=run.sample_rate, mono=True)
            feats = np.asarray(features.extract(audio, run).feats, dtype=np.float32)
            pcm = np.asarray(audio.pcm, dtype=np.float32)
            out.append((prompt_stem(p), feats, pcm, run.master_clock_hz, run.frame_rate_hz))
        except Exception as exc:  # noqa: BLE001 — one bad clip must not kill the run
            print(f"[skip] {p}: {exc}", file=sys.stderr, flush=True)
        if i % log_every == 0 or i == total:
            print(f"  loaded {i}/{total}", flush=True)
    print(f"Load done: {len(out)} ok, {total - len(out)} skipped.", flush=True)
    return out


def _crop_real(
    sample: RealSample, window: int | None, rng: np.random.Generator
) -> RealSample:
    """Random ``window``-frame slice of features with the time-aligned PCM span (full clip if short)."""
    stem, feats, pcm, mclk, fr = sample
    t = feats.shape[0]
    if window is None or t <= window:
        return sample
    s = int(rng.integers(0, t - window + 1))
    spp = pcm.shape[0] / max(1, t)                       # samples per frame
    a, b = int(round(s * spp)), int(round((s + window) * spp))
    return stem, feats[s:s + window], pcm[a:b], mclk, fr


def _lr_at(step: int, base_lr: float, max_steps: int, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def real_forward(
    net: ReversePlayer,
    emulator: DiffAyEmulator,
    samples: list[RealSample],
    *,
    device: str,
    weights: RewardWeights,
    tau: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Analysis-by-synthesis against **real** audio → ``(total_loss, parts)``.

    Lengths vary per clip (no shared YM grid), so chroma/onset/spectral terms are computed per clip
    against that clip's own PCM and averaged; the jitter term averages over the relaxed controls.
    """
    net.train()
    chroma_t = torch.zeros((), device=device)
    onset_t = torch.zeros((), device=device)
    spec_t = torch.zeros((), device=device)
    jitter_t = torch.zeros((), device=device)

    for _stem, feats, pcm, mclk, fr in samples:
        x = torch.from_numpy(np.ascontiguousarray(feats.T)).unsqueeze(0).to(device)  # (1, dim, T)
        heads = net(x)
        controls = controls_from_heads(heads, float(mclk), tau=tau)
        recon = emulator.render(controls.select(0), mclk, fr)            # (N,) differentiable
        target = torch.from_numpy(pcm).to(device)                        # (M,) real audio
        sr = emulator.render_sr
        if weights.chroma > 0.0:
            chroma_t = chroma_t + chroma_loss(recon, target, sr)
        if weights.onset > 0.0:
            onset_t = onset_t + onset_loss(recon, target, sr)
        if weights.spectral > 0.0:
            spec_t = spec_t + multiscale_stft_loss(recon, target)
        jitter_t = jitter_t + jitter_penalty(controls)

    k = max(1, len(samples))
    chroma_t, onset_t, spec_t, jitter_t = chroma_t / k, onset_t / k, spec_t / k, jitter_t / k
    total = (weights.chroma * chroma_t + weights.onset * onset_t
             + weights.spectral * spec_t + weights.jitter * jitter_t)
    parts = {"jitter": float(jitter_t.detach())}
    if weights.chroma > 0.0:
        parts["chroma"] = float(chroma_t.detach())
    if weights.onset > 0.0:
        parts["onset"] = float(onset_t.detach())
    if weights.spectral > 0.0:
        parts["spectral"] = float(spec_t.detach())
    return total, parts


def _score_heldout(paths: list[str], checkpoint: str, hidden: int, base: RunConfig) -> dict:
    """Run the *deployed* rl core (this checkpoint) over held-out clips → aggregate eval metrics."""
    from ..config import RunConfig as _RC
    from ..eval import aggregate, evaluate_audio

    cfg = _RC(
        core="rl",
        master_clock_hz=base.master_clock_hz,
        frame_rate_hz=base.frame_rate_hz,
        sample_rate=base.sample_rate,
        feat_kind=base.feat_kind,
        extra={"checkpoint": checkpoint, "hidden": hidden},
    )
    results = [evaluate_audio(p, cfg) for p in paths]
    return aggregate(results)


def _checkpoint_path(train_cfg: TrainConfig) -> str:
    return train_cfg.run.extra.get("checkpoint") or os.path.join(
        train_cfg.cache_dir or ".cache", "reward_rl_real.pt"
    )


def train_real(
    train_cfg: TrainConfig,
    audio_paths: list[str],
    *,
    init_checkpoint: str | None = None,
    device: str | None = None,
    window: int | None = 512,
    weights: RewardWeights | None = None,
    tau: float = 1.0,
    heldout_frac: float = 0.3,
    max_partials: int = 24,
    oversample: int = 2,
    log_every: int = 25,
) -> str:
    """Split → real-audio analysis-by-synthesis train → score held-out (Stage-1 go/no-go). Returns ckpt."""
    if not audio_paths:
        raise ValueError("train_real needs at least one audio file")
    device = device or ("cuda" if (train_cfg.run.use_gpu and torch.cuda.is_available()) else "cpu")
    run: RunConfig = train_cfg.run
    weights = weights or RewardWeights(spectral=0.0, jitter=0.0, chroma=5.0, onset=1.0)

    train_paths, heldout_paths = split_by_stem(audio_paths, heldout_frac, run.seed)
    print(
        f"Split {len(audio_paths)} clips → {len(train_paths)} train / {len(heldout_paths)} held-out "
        f"(by prompt stem, frac={heldout_frac:.2f}, seed={run.seed})",
        flush=True,
    )
    if not train_paths:
        raise ValueError("split left no training clips — lower --heldout-frac")

    samples = load_real_samples(train_paths, run)
    if not samples:
        raise ValueError("No usable audio after decoding (all failed)")
    dim = samples[0][1].shape[1]

    torch.manual_seed(run.seed)
    rng = np.random.default_rng(run.seed)
    hidden = int(run.extra.get("hidden", 128))
    net = ReversePlayer(in_dim=dim, hidden=hidden).to(device)
    if init_checkpoint:
        state = torch.load(init_checkpoint, map_location="cpu")
        net.load_state_dict(state.get("model", state))
        print(f"Initialised from checkpoint {init_checkpoint}")
    opt = torch.optim.AdamW(net.parameters(), lr=train_cfg.lr, weight_decay=0.0)

    emulator = DiffAyEmulator(
        render_sr=run.sample_rate, oversample=oversample, max_partials=max_partials
    ).to(device)

    bs = max(1, min(train_cfg.batch_size, len(samples)))
    base_lr = train_cfg.lr
    warmup = max(1, min(200, train_cfg.max_steps // 10))
    print(
        f"Real-audio training on {device}: {len(samples)} clips | batch {bs} "
        f"| window {window or 'full'} | {train_cfg.max_steps} steps "
        f"| lr {base_lr:.1e} (warmup {warmup}, cosine) "
        f"| w_spec {weights.spectral:.2f} w_jit {weights.jitter:.3f} "
        f"w_chroma {weights.chroma:.2f} w_onset {weights.onset:.2f} tau {tau:.2f} "
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
        batch = [_crop_real(samples[i], window, rng) for i in idx]
        total, parts = real_forward(net, emulator, batch, device=device, weights=weights, tau=tau)
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
    print(f"Saved real-audio checkpoint → {out}")

    manifest = out + ".heldout.json"
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"train": train_paths, "heldout": heldout_paths}, fh, indent=2)
    print(f"Wrote split manifest → {manifest}")

    if heldout_paths:
        print("\n=== Stage-1 go/no-go: held-out chroma/onset (unseen clips) ===", flush=True)
        new = _score_heldout(heldout_paths, out, hidden, run)
        line_new = (f"  NEW  ({os.path.basename(out)}): "
                    f"chroma={new['chroma_sim']:.4f} onset={new['onset_sim']:.4f} "
                    f"legality={new['legality_rate']:.3f}")
        if init_checkpoint:
            base_agg = _score_heldout(heldout_paths, init_checkpoint, hidden, run)
            print(f"  INIT ({os.path.basename(init_checkpoint)}): "
                  f"chroma={base_agg['chroma_sim']:.4f} onset={base_agg['onset_sim']:.4f} "
                  f"legality={base_agg['legality_rate']:.3f}", flush=True)
            print(line_new, flush=True)
            d_chroma = new["chroma_sim"] - base_agg["chroma_sim"]
            d_onset = new["onset_sim"] - base_agg["onset_sim"]
            verdict = "GO ✅ (real-audio reward lifts held-out chroma)" if d_chroma > 0 \
                else "NO-GO ❌ (no held-out chroma gain — do not scale the corpus)"
            print(f"  Δ chroma={d_chroma:+.4f}  Δ onset={d_onset:+.4f}  →  {verdict}", flush=True)
        else:
            print(line_new, flush=True)

    return out


__all__ = [
    "RealSample",
    "find_audio_files",
    "prompt_stem",
    "split_by_stem",
    "load_real_samples",
    "real_forward",
    "train_real",
]
