"""Supervised warm-start loss (design A.5): regress continuous heads, classify gates/categoricals.

Operates on the reverse player's **raw** head outputs and the padded target batch from
``warmstart``. Continuous heads are decoded with the *same* transforms inference uses (so train and
inference agree), then compared in natural units (semitones, dB, log-Hz, probability). Gate heads
use BCE-with-logits; ``env_shape`` uses cross-entropy. Every term is masked: pitch only where the
tone is on, volume only where a voice is audible and not envelope-controlled, and the envelope
rate/shape only on frames that (re)write R13.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.policy.spec import (
    ENV_RATE_FLOOR_HZ,
    PITCH_CENTER,
    PITCH_SPAN,
    VOL_CEIL_DB,
    VOL_FLOOR_DB,
)


@dataclass(frozen=True)
class WarmstartWeights:
    pitch: float = 1.0
    volume: float = 1.0
    tone: float = 1.0
    noise: float = 1.0
    env_use: float = 1.0
    noise_pitch: float = 0.5
    env_rate: float = 0.5
    env_shape: float = 0.5
    env_retrig: float = 0.5


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (mask * (pred - target) ** 2).sum() / denom


def _masked_bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (mask * bce).sum() / mask.sum().clamp(min=1.0)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")  # (B, T)
    return (mask * ce).sum() / mask.sum().clamp(min=1.0)


def warmstart_loss(
    heads: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    pad_mask: torch.Tensor,
    weights: WarmstartWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``(total_loss, parts)`` where ``parts`` are detached scalars for logging."""
    w = weights or WarmstartWeights()
    pad1 = pad_mask.unsqueeze(1)                         # (B, 1, T) → broadcast over voices
    tone_t = targets["tone"]
    audible = torch.maximum(tone_t, targets["noise"])    # voice contributes sound

    # Continuous heads decoded exactly as at inference.
    dpitch = PITCH_CENTER + PITCH_SPAN * torch.tanh(heads["pitch"])
    dvol = VOL_FLOOR_DB + (VOL_CEIL_DB - VOL_FLOOR_DB) * torch.sigmoid(heads["volume"])
    drate = ENV_RATE_FLOOR_HZ + F.softplus(heads["env_rate"].squeeze(1))
    np_pred = torch.sigmoid(heads["noise_pitch"].squeeze(1))

    pitch_mask = tone_t * pad1
    vol_mask = audible * (1.0 - targets["env_use"]) * pad1
    env_active = targets["env_retrig"] * pad_mask        # (B, T)

    parts: dict[str, torch.Tensor] = {
        "pitch": _masked_mse(dpitch, targets["pitch"], pitch_mask),
        "volume": _masked_mse(dvol, targets["volume"], vol_mask),
        "tone": _masked_bce(heads["tone_logit"], tone_t, pad1.expand_as(tone_t)),
        "noise": _masked_bce(heads["noise_logit"], targets["noise"], pad1.expand_as(tone_t)),
        "env_use": _masked_bce(heads["env_use_logit"], targets["env_use"], pad1.expand_as(tone_t)),
        "noise_pitch": _masked_mse(np_pred, targets["noise_pitch"], pad_mask),
        "env_rate": _masked_mse(torch.log(drate), torch.log(targets["env_rate"]), env_active),
        "env_shape": _masked_ce(heads["env_shape"], targets["env_shape"], env_active),
        "env_retrig": _masked_bce(heads["env_retrig"].squeeze(1), targets["env_retrig"], pad_mask),
    }
    total = (
        w.pitch * parts["pitch"] + w.volume * parts["volume"] + w.tone * parts["tone"]
        + w.noise * parts["noise"] + w.env_use * parts["env_use"]
        + w.noise_pitch * parts["noise_pitch"] + w.env_rate * parts["env_rate"]
        + w.env_shape * parts["env_shape"] + w.env_retrig * parts["env_retrig"]
    )
    return total, {k: float(v.detach()) for k, v in parts.items()}
