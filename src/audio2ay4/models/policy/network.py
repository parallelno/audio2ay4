"""Plan A reverse player ``E`` — the learned core's neural body.

A non-causal dilated **temporal convolutional network** (TCN) over the 50 fps feature grid that
emits, per frame, the raw head activations for an ``AYStateFrame`` (§A.2 of
design/plan-a-reinforcement-learning.md). It is non-causal on purpose: conversion is offline, so
``E`` may see the whole track (symmetric padding ⇒ output length == input length, any ``T``).

This module only produces **raw** head tensors; mapping them to a smooth, legal ``AYState`` lives
in :mod:`audio2ay4.models.policy.core` so the network stays a pure, differentiable function (ready
for the supervised warm-start in A2 and the reward training in A4).

Requires the optional ``neural`` extra (torch). It is imported lazily — only when the ``rl`` core
is requested — so the deterministic core stays numpy-only importable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .spec import N_ENV_SHAPES, N_PITCH_BINS, N_VOICES, N_VOL_LEVELS


class _ResidualBlock(nn.Module):
    """Non-causal dilated residual block (symmetric padding keeps the time length fixed)."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, groups: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # symmetric ⇒ non-causal, length-preserving
        g = groups if channels % groups == 0 else 1
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(g, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.act(self.norm2(self.conv2(y)))
        y = self.drop(y)
        return x + y


class ReversePlayer(nn.Module):
    """features ``(B, in_dim, T)`` → dict of per-frame head tensors (all length ``T``).

    Heads (each a 1×1 conv over the temporal embedding):
      * ``pitch_logits``  (B, 3, K, T) categorical over K semitone bins, per voice
      * ``volume_logits`` (B, 3, L, T) categorical over L DAC levels, per voice
      * ``tone_logit``    (B, 3, T)   Bernoulli gate, per voice
      * ``noise_logit``   (B, 3, T)   Bernoulli gate, per voice
      * ``env_use_logit`` (B, 3, T)   Bernoulli gate, per voice
      * ``noise_pitch``   (B, 1, T)   continuous, shared
      * ``env_rate``      (B, 1, T)   continuous, shared
      * ``env_shape``     (B, 16, T)  categorical logits, shared
      * ``env_retrig``    (B, 1, T)   Bernoulli gate, shared
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.in_proj = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden, dilation=d, kernel_size=kernel_size, dropout=dropout)
            for d in dilations
        )
        # Per-frame heads.
        self.head_pitch = nn.Conv1d(hidden, N_VOICES * N_PITCH_BINS, 1)
        self.head_volume = nn.Conv1d(hidden, N_VOICES * N_VOL_LEVELS, 1)
        self.head_tone = nn.Conv1d(hidden, N_VOICES, 1)
        self.head_noise = nn.Conv1d(hidden, N_VOICES, 1)
        self.head_env_use = nn.Conv1d(hidden, N_VOICES, 1)
        self.head_noise_pitch = nn.Conv1d(hidden, 1, 1)
        self.head_env_rate = nn.Conv1d(hidden, 1, 1)
        self.head_env_shape = nn.Conv1d(hidden, N_ENV_SHAPES, 1)
        self.head_env_retrig = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"expected (B, in_dim, T), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"in_dim mismatch: model={self.in_dim}, input={x.shape[1]}")
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        pitch = self.head_pitch(h)                                          # (B, V*K, T)
        pitch_logits = pitch.view(pitch.shape[0], N_VOICES, N_PITCH_BINS, pitch.shape[-1])
        volume = self.head_volume(h)                                        # (B, V*L, T)
        volume_logits = volume.view(volume.shape[0], N_VOICES, N_VOL_LEVELS, volume.shape[-1])
        return {
            "pitch_logits": pitch_logits,
            "volume_logits": volume_logits,
            "tone_logit": self.head_tone(h),
            "noise_logit": self.head_noise(h),
            "env_use_logit": self.head_env_use(h),
            "noise_pitch": self.head_noise_pitch(h),
            "env_rate": self.head_env_rate(h),
            "env_shape": self.head_env_shape(h),
            "env_retrig": self.head_env_retrig(h),
        }
