"""Supervised warm-start loss (design A.5): regress continuous heads, classify gates/categoricals.

Operates on the reverse player's **raw** head outputs and the padded target batch from
``warmstart``. Continuous heads are decoded with the *same* transforms inference uses (so train and
inference agree), then compared in natural units (semitones, dB, log-Hz, probability). Gate heads
use BCE-with-logits; ``env_shape`` uses cross-entropy. Every term is masked: pitch only where the
tone is on, volume only where a voice is audible and not envelope-controlled, and the envelope
rate/shape only on frames that (re)write R13.

The three AY tone channels render to identical timbres, so the corpus's A/B/C voice ordering is not
recoverable from the mixed audio. The per-voice heads are therefore scored **permutation-invariantly**
(utterance-level PIT): per tune we pick the predicted→target voice assignment with the lowest cost
before reducing, so an audio-equivalent relabelling is never penalised.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.policy.spec import (
    ENV_RATE_FLOOR_HZ,
    N_VOICES,
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


_PERMS = tuple(itertools.permutations(range(N_VOICES)))  # 6 voice assignments (V = 3)


def _best_voice_perm(
    heads: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    pad_mask: torch.Tensor,
    w: WarmstartWeights,
) -> torch.Tensor:
    """Pick, per tune, the predicted→target voice assignment with the lowest combined cost.

    The three AY tone channels render to identical timbres, so the corpus's arbitrary A/B/C voice
    ordering is not recoverable from the mixed audio. Forcing it makes per-voice pitch partly
    unlearnable. We instead score all ``V! = 6`` assignments (pitch CE + volume MSE + gate BCEs,
    summed over the sequence) and return the best as a ``(B, V)`` index of target voices.
    """
    b, v, k, t = heads["pitch_logits"].shape
    pad_ij = pad_mask.view(b, 1, 1, t)

    logp = F.log_softmax(heads["pitch_logits"], dim=2)            # (B, Vp, K, T)
    bins = targets["pitch_bin"]                                   # (B, Vt, T)
    idx = bins.unsqueeze(1).unsqueeze(3).expand(b, v, v, 1, t)    # (B, Vp, Vt, 1, T)
    ce = -logp.unsqueeze(2).expand(b, v, v, k, t).gather(3, idx).squeeze(3)  # (B, Vp, Vt, T)

    vlogp = F.log_softmax(heads["volume_logits"], dim=2)          # (B, Vp, L, T)
    levels = targets["volume_level"]                              # (B, Vt, T)
    lvl = heads["volume_logits"].shape[2]
    vidx = levels.unsqueeze(1).unsqueeze(3).expand(b, v, v, 1, t)  # (B, Vp, Vt, 1, T)
    vol_ce = -vlogp.unsqueeze(2).expand(b, v, v, lvl, t).gather(3, vidx).squeeze(3)  # (B,Vp,Vt,T)

    def _pair_bce(logit_key: str, tgt_key: str) -> torch.Tensor:
        lo = heads[logit_key].unsqueeze(2).expand(b, v, v, t)
        tg = targets[tgt_key].unsqueeze(1).expand(b, v, v, t)
        return F.binary_cross_entropy_with_logits(lo, tg, reduction="none")

    tone_j = targets["tone"].unsqueeze(1)                                   # (B, 1, Vt, T)
    audible_j = torch.maximum(targets["tone"], targets["noise"]).unsqueeze(1)
    pitch_mask = tone_j * pad_ij
    vol_mask = audible_j * (1.0 - targets["env_use"].unsqueeze(1)) * pad_ij

    cost = (
        w.pitch * pitch_mask * ce
        + w.volume * vol_mask * vol_ce
        + pad_ij * (w.tone * _pair_bce("tone_logit", "tone")
                    + w.noise * _pair_bce("noise_logit", "noise")
                    + w.env_use * _pair_bce("env_use_logit", "env_use"))
    )
    m = cost.sum(dim=3)                                                     # (B, Vp, Vt)

    perms = torch.tensor(_PERMS, device=m.device)                          # (P, V)
    rows = torch.arange(v, device=m.device)
    perm_costs = torch.stack(
        [m[:, rows, perms[p]].sum(dim=1) for p in range(perms.shape[0])], dim=0
    )                                                                      # (P, B)
    return perms[perm_costs.argmin(dim=0)]  # (B, V) target voice indices


def warmstart_loss(
    heads: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    pad_mask: torch.Tensor,
    weights: WarmstartWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``(total_loss, parts)`` where ``parts`` are detached scalars for logging."""
    w = weights or WarmstartWeights()

    # Permutation-invariant voice assignment: relabel target voices to the best match before
    # scoring, so audio-equivalent channel orderings are not penalised (selection has no grad).
    with torch.no_grad():
        perm = _best_voice_perm(heads, targets, pad_mask, w)         # (B, V) target voice indices
    gather_idx = perm.unsqueeze(-1).expand(-1, -1, pad_mask.shape[1])  # (B, V, T)
    pt = dict(targets)
    for key in ("pitch_bin", "volume_level", "tone", "noise", "env_use"):
        pt[key] = targets[key].gather(1, gather_idx)

    pad1 = pad_mask.unsqueeze(1)                         # (B, 1, T) → broadcast over voices
    tone_t = pt["tone"]
    audible = torch.maximum(tone_t, pt["noise"])         # voice contributes sound

    # Continuous shared heads decoded exactly as at inference; pitch and volume are per-voice
    # classifications (over semitone bins / DAC levels).
    drate = ENV_RATE_FLOOR_HZ + F.softplus(heads["env_rate"].squeeze(1))
    np_pred = torch.sigmoid(heads["noise_pitch"].squeeze(1))

    pitch_mask = tone_t * pad1
    vol_mask = audible * (1.0 - pt["env_use"]) * pad1
    env_active = targets["env_retrig"] * pad_mask        # (B, T) — shared head, no permutation

    # Cross-entropy over pitch bins / volume levels: logits (B, V, C, T) → (B, C, V, T).
    pitch_ce = F.cross_entropy(
        heads["pitch_logits"].permute(0, 2, 1, 3), pt["pitch_bin"], reduction="none"
    )                                                    # (B, V, T)
    volume_ce = F.cross_entropy(
        heads["volume_logits"].permute(0, 2, 1, 3), pt["volume_level"], reduction="none"
    )                                                    # (B, V, T)

    parts: dict[str, torch.Tensor] = {
        "pitch": (pitch_mask * pitch_ce).sum() / pitch_mask.sum().clamp(min=1.0),
        "volume": (vol_mask * volume_ce).sum() / vol_mask.sum().clamp(min=1.0),
        "tone": _masked_bce(heads["tone_logit"], tone_t, pad1.expand_as(tone_t)),
        "noise": _masked_bce(heads["noise_logit"], pt["noise"], pad1.expand_as(tone_t)),
        "env_use": _masked_bce(heads["env_use_logit"], pt["env_use"], pad1.expand_as(tone_t)),
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
