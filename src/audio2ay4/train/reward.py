"""Perceptual reconstruction reward / loss for Regime-1 analysis-by-synthesis (design A.4).

Computed between the **target** audio (the corpus performance, rendered by the differentiable
emulator from its ground-truth registers) and the **reconstruction** (the same emulator driven by
the relaxed policy controls). Rendering both through the *same* differentiable emulator isolates the
learning signal to the controls and gives a true zero floor when they match (the §A.11 overfit-one-
track canary needs that floor; the trusted-vs-twin emulator gap is A5's real-audio concern).

Terms (design A.4):
  * ``multiscale_stft`` — multi-resolution STFT L1 (linear + log magnitude), the coarse-to-fine
    spectral term that rewards pitch / rhythm / harmony agreement.
  * ``jitter`` — frame-to-frame period / level thrash on the emitted controls; the learned
    replacement for ``audio2ay3``'s hand-tuned hysteresis (stability becomes part of the objective).

The timbre-invariant embedding term (w2, CLAP-style) and the corpus idiomatic prior (A.6) are
deliberately out of scope here — they belong to the real/augmented-audio phase (A5) and need a heavy
external encoder. The interface leaves room for them via additive weighted terms.

Torch-only and training-time; imported lazily, never by the numpy core.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..models.policy.relax import BatchControls

# Multi-resolution STFT window sizes (coarse → fine), the standard DDSP ladder.
_DEFAULT_FFTS: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)


@dataclass(frozen=True)
class RewardWeights:
    """Relative weights of the reconstruction reward terms (design A.4).

    ``chroma`` (melody / harmony, pitch-class) and ``onset`` (rhythm) are the timbre-invariant terms
    that the human-validated ``eval.metrics`` showed actually track perceived quality — unlike raw
    ``spectral`` magnitude, which a steady beep can game. Default 0 keeps legacy behaviour; the
    melody-first retrain turns them up and ``spectral`` / ``jitter`` down.
    """

    spectral: float = 1.0
    jitter: float = 0.02
    chroma: float = 0.0
    onset: float = 0.0


def multiscale_stft_loss(
    recon: Tensor,
    target: Tensor,
    *,
    ffts: tuple[int, ...] = _DEFAULT_FFTS,
) -> Tensor:
    """Multi-resolution STFT distance (linear + ``log1p`` magnitude L1), averaged over scales.

    Accepts ``(N,)`` or ``(B, N)`` mono signals; the two are truncated to a common length. Returns a
    scalar (0 ⇒ spectrally identical). Scales larger than the signal are skipped.

    The log term uses ``log1p`` (matching ``eval.metrics``), not ``log(x + eps)``: it is gentle on
    the many near-silent bins (``log1p(0) = 0``) instead of dominating the loss with a huge negative
    floor there, which keeps the analysis-by-synthesis optimisation well-conditioned.
    """
    if recon.dim() == 1:
        recon = recon.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    n = min(recon.shape[-1], target.shape[-1])
    recon, target = recon[..., :n], target[..., :n]

    total = recon.new_zeros(())
    used = 0
    for n_fft in ffts:
        if n < n_fft:
            continue
        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=recon.device, dtype=recon.dtype)
        mr = torch.stft(recon, n_fft, hop, window=win, return_complex=True).abs()
        mt = torch.stft(target, n_fft, hop, window=win, return_complex=True).abs()
        lin = (mr - mt).abs().mean()
        log = (torch.log1p(mr) - torch.log1p(mt)).abs().mean()
        total = total + lin + log
        used += 1
    if used == 0:
        return total
    return total / used


def jitter_penalty(controls: BatchControls) -> Tensor:
    """Frame-to-frame thrash on the emitted controls (relative tone-period + DAC-level change).

    Penalises zipper noise / register churn the way ``audio2ay3``'s hysteresis did, but as part of
    the objective. Tone-period change is taken relatively (a fixed Hz wobble matters more at high
    pitch); level change is absolute in DAC steps. Weighted by audibility so silent voices are free.
    """
    tp = controls.tone_period                                   # (B, T, 3)
    lvl = controls.level                                        # (B, T, 3)
    audible = (controls.tone_gate * (1.0 - controls.use_env) + controls.use_env).clamp(0.0, 1.0)
    w = audible[:, 1:] * audible[:, :-1]                        # both frames audible
    if tp.shape[1] < 2:
        return tp.new_zeros(())
    d_tp = (tp[:, 1:] - tp[:, :-1]).abs() / tp[:, :-1].clamp(min=1.0)
    d_lvl = (lvl[:, 1:] - lvl[:, :-1]).abs() / 15.0
    denom = w.sum().clamp(min=1.0)
    return ((w * (d_tp + d_lvl)).sum()) / denom


_A4_HZ = 440.0


def _stft_mag(x: Tensor, n_fft: int, hop: int) -> Tensor:
    """Linear-magnitude STFT of ``(B, N)`` (or ``(N,)``) → ``(B, F, T)``."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft, hop, window=win, return_complex=True).abs()


def _chroma_filterbank(n_fft: int, sr: int, fmin: float, fmax: float,
                       device: torch.device, dtype: torch.dtype) -> Tensor:
    """Constant ``(12, F)`` matrix folding each in-range rfft bin onto its pitch class."""
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(device=device, dtype=dtype)
    fb = torch.zeros(12, freqs.shape[0], device=device, dtype=dtype)
    mask = (freqs >= fmin) & (freqs <= fmax)
    midi = 69.0 + 12.0 * torch.log2(freqs.clamp(min=1e-6) / _A4_HZ)
    pc = torch.round(midi).long() % 12
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    fb[pc[idx], idx] = 1.0
    return fb


def chroma_loss(
    recon: Tensor,
    target: Tensor,
    sample_rate: int,
    *,
    n_fft: int = 4096,
    hop: int = 1024,
    fmin: float = 65.0,
    fmax: float = 2093.0,
) -> Tensor:
    """``1 - (target-energy-weighted mean per-frame chroma cosine)`` — the melody / harmony loss.

    Differentiable torch twin of ``eval.metrics.chroma_similarity`` (same params), so we *train on
    what we measure*: timbre-invariant pitch-class agreement. Frames are weighted by the target's
    chroma energy so silent gaps don't dilute the signal. Returns a scalar in ``[0, ~1]``.
    """
    if recon.dim() == 1:
        recon = recon.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    n = min(recon.shape[-1], target.shape[-1])
    if n < n_fft:
        return recon.new_zeros(())
    recon, target = recon[..., :n], target[..., :n]
    mr = _stft_mag(recon, n_fft, hop)                                   # (B, F, T)
    mt = _stft_mag(target, n_fft, hop)
    fb = _chroma_filterbank(n_fft, sample_rate, fmin, fmax, recon.device, recon.dtype)
    cr = torch.einsum("cf,bft->bct", fb, mr)                            # (B, 12, T)
    ct = torch.einsum("cf,bft->bct", fb, mt)
    cr_n = cr / cr.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ct_n = ct / ct.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos = (cr_n * ct_n).sum(dim=1)                                      # (B, T)
    w = ct.norm(dim=1)                                                  # weight by target energy
    sim = (cos * w).sum() / w.sum().clamp(min=1e-6)
    return 1.0 - sim


def onset_loss(
    recon: Tensor,
    target: Tensor,
    sample_rate: int,  # noqa: ARG001 — kept for signature symmetry with chroma_loss / eval
    *,
    n_fft: int = 2048,
    hop: int = 512,
) -> Tensor:
    """``1 - Pearson(onset_env_recon, onset_env_target)`` — the rhythm / timing loss.

    Differentiable torch twin of ``eval.metrics.onset_similarity``: positive spectral flux gives a
    per-frame onset envelope, correlated over time. Rewards landing transients (drums) with the
    target. Returns a scalar in ``[0, 2]`` (0 ⇒ perfectly correlated).
    """
    if recon.dim() == 1:
        recon = recon.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    n = min(recon.shape[-1], target.shape[-1])
    if n < 2 * n_fft:
        return recon.new_zeros(())
    recon, target = recon[..., :n], target[..., :n]
    fr = torch.relu(_stft_mag(recon, n_fft, hop).diff(dim=-1)).sum(dim=1)   # (B, T-1)
    ft = torch.relu(_stft_mag(target, n_fft, hop).diff(dim=-1)).sum(dim=1)
    er = fr - fr.mean(dim=1, keepdim=True)
    et = ft - ft.mean(dim=1, keepdim=True)
    num = (er * et).sum(dim=1)
    den = (er.norm(dim=1) * et.norm(dim=1)).clamp(min=1e-9)
    corr = (num / den).mean()
    return 1.0 - corr


def reward_loss(
    recon: Tensor,
    target: Tensor,
    controls: BatchControls,
    weights: RewardWeights | None = None,
    *,
    sample_rate: int = 44_100,
    ffts: tuple[int, ...] = _DEFAULT_FFTS,
) -> tuple[Tensor, dict[str, float]]:
    """Total reconstruction loss + detached scalar parts for logging."""
    w = weights or RewardWeights()
    spec = multiscale_stft_loss(recon, target, ffts=ffts)
    jit = jitter_penalty(controls)
    total = w.spectral * spec + w.jitter * jit
    parts = {"spectral": float(spec.detach()), "jitter": float(jit.detach())}
    if w.chroma > 0.0:
        chr_ = chroma_loss(recon, target, sample_rate)
        total = total + w.chroma * chr_
        parts["chroma"] = float(chr_.detach())
    if w.onset > 0.0:
        ons = onset_loss(recon, target, sample_rate)
        total = total + w.onset * ons
        parts["onset"] = float(ons.detach())
    return total, parts


__all__ = [
    "RewardWeights",
    "multiscale_stft_loss",
    "jitter_penalty",
    "chroma_loss",
    "onset_loss",
    "reward_loss",
]
