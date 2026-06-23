"""Differentiable relaxation: reverse-player head logits → continuous controls (A.3, Regime 1).

Bridges ``ReversePlayer``'s classification / Bernoulli heads to the differentiable emulator's
continuous :class:`~audio2ay4.chip.diff.DiffControls` space so gradients flow
``E → controls → DiffAyEmulator → reconstruction loss`` (analysis-by-synthesis).

Categorical heads (pitch, volume, noise period, envelope rate) are relaxed with a temperature
**soft-argmax** — the differentiable expected value over the softmax. For the peaked distributions a
warm-started ``E`` produces this is low-variance and effectively exact, and it needs no Gumbel
sampling noise. Gates (tone / noise / use-env / retrigger) pass through their sigmoid probability.
The envelope shape nibble has no smooth ordering (and the emulator's shape decomposition is
non-smooth), so it is taken as a hard arg-max — its gradient would be meaningless anyway.

This module is torch-only and training-time: it is imported lazily, never by the numpy core.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ...chip.diff import DiffControls
from ...repr.compile import EP_MAX, EP_MIN, NP_MIN, TP_MIN
from .spec import PITCH_BIN_WIDTH, PITCH_MIN, bin_to_env_rate


@dataclass
class BatchControls:
    """Batched, differentiable controls (leading ``(B, T, ...)`` axis); ``select`` → one track."""

    tone_period: Tensor   # (B, T, 3)
    noise_period: Tensor  # (B, T)
    env_period: Tensor    # (B, T)
    tone_gate: Tensor     # (B, T, 3)
    noise_gate: Tensor    # (B, T, 3)
    level: Tensor         # (B, T, 3)
    use_env: Tensor       # (B, T, 3)
    env_shape: Tensor     # (B, T) long
    env_retrig: Tensor    # (B, T)

    def select(self, b: int) -> DiffControls:
        """Slice batch item ``b`` into a per-track :class:`DiffControls` (leading axis ``T``)."""
        return DiffControls(
            tone_period=self.tone_period[b],
            noise_period=self.noise_period[b],
            env_period=self.env_period[b],
            tone_gate=self.tone_gate[b],
            noise_gate=self.noise_gate[b],
            level=self.level[b],
            use_env=self.use_env[b],
            env_shape=self.env_shape[b],
            env_retrig=self.env_retrig[b],
        )


@functools.lru_cache(maxsize=8)
def _env_log_rates(n_bins: int) -> tuple[float, ...]:
    """Log of each envelope-rate bin centre (Hz), for the soft-argmax over rate classes."""
    import math

    return tuple(math.log(bin_to_env_rate(b)) for b in range(n_bins))


def controls_from_heads(
    heads: dict[str, Tensor],
    master_clock: float,
    *,
    tau: float = 1.0,
) -> BatchControls:
    """Relax raw network heads into differentiable :class:`BatchControls`.

    ``master_clock`` is the AY clock the controls are interpreted at: it converts the soft pitch /
    envelope-rate expectations into period registers so that, when the emulator renders at the same
    clock, the synthesised frequency equals the predicted one (the conversion round-trips).
    """
    pitch_logits = heads["pitch_logits"]                       # (B, 3, K, T)
    device, dtype = pitch_logits.device, pitch_logits.dtype
    k = pitch_logits.shape[2]

    # --- pitch → tone period (soft expected semitone → Hz → TP) -----------------------------
    bin_semi = PITCH_MIN + torch.arange(k, device=device, dtype=dtype) * PITCH_BIN_WIDTH
    exp_semi = (F.softmax(pitch_logits / tau, dim=2) * bin_semi.view(1, 1, k, 1)).sum(dim=2)
    hz = 440.0 * torch.pow(2.0, (exp_semi - 69.0) / 12.0)       # (B, 3, T)
    tone_period = (master_clock / (16.0 * hz)).clamp(min=float(TP_MIN)).permute(0, 2, 1)

    # --- volume → continuous DAC level (soft expected level in [0, L-1]) ---------------------
    vol_logits = heads["volume_logits"]                        # (B, 3, L, T)
    n_lvl = vol_logits.shape[2]
    levels = torch.arange(n_lvl, device=device, dtype=dtype)
    level = (F.softmax(vol_logits / tau, dim=2) * levels.view(1, 1, n_lvl, 1)).sum(dim=2)
    level = level.permute(0, 2, 1)                             # (B, T, 3)

    # --- gates (Bernoulli probabilities) -----------------------------------------------------
    tone_gate = torch.sigmoid(heads["tone_logit"]).permute(0, 2, 1)      # (B, T, 3)
    noise_gate = torch.sigmoid(heads["noise_logit"]).permute(0, 2, 1)
    use_env = torch.sigmoid(heads["env_use_logit"]).permute(0, 2, 1)

    # --- noise period (soft expected 5-bit period level) -------------------------------------
    np_logits = heads["noise_pitch_logits"]                    # (B, P, T)
    n_np = np_logits.shape[1]
    np_levels = torch.arange(n_np, device=device, dtype=dtype)
    noise_period = (F.softmax(np_logits / tau, dim=1) * np_levels.view(1, n_np, 1)).sum(dim=1)
    noise_period = noise_period + float(NP_MIN)                # (B, T)

    # --- envelope period (soft expected log-rate → EP) ---------------------------------------
    er_logits = heads["env_rate_logits"]                       # (B, R, T)
    log_rate = torch.tensor(_env_log_rates(er_logits.shape[1]), device=device, dtype=dtype)
    exp_log = (F.softmax(er_logits / tau, dim=1) * log_rate.view(1, -1, 1)).sum(dim=1)
    rate_hz = torch.exp(exp_log)                               # (B, T)
    env_period = (master_clock / (256.0 * rate_hz)).clamp(float(EP_MIN), float(EP_MAX))

    # --- envelope shape (hard arg-max; no smooth ordering) + retrigger gate ------------------
    env_shape = heads["env_shape"].argmax(dim=1)               # (B, T) long
    env_retrig = torch.sigmoid(heads["env_retrig"].squeeze(1))  # (B, T)

    return BatchControls(
        tone_period=tone_period,
        noise_period=noise_period,
        env_period=env_period,
        tone_gate=tone_gate,
        noise_gate=noise_gate,
        level=level,
        use_env=use_env,
        env_shape=env_shape,
        env_retrig=env_retrig,
    )
