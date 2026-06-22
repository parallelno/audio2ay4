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

# Per-frame head widths. Three tone voices share the same per-voice layout; the global block
# models the single shared envelope + single shared noise generator exactly once.
N_VOICES = 3
N_ENV_SHAPES = 16


class _ResidualBlock(nn.Module):
    """Non-causal dilated residual block (symmetric padding keeps the time length fixed)."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, groups: int = 8) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # symmetric ⇒ non-causal, length-preserving
        g = groups if channels % groups == 0 else 1
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(g, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.act(self.norm2(self.conv2(y)))
        return x + y


class ReversePlayer(nn.Module):
    """features ``(B, in_dim, T)`` → dict of per-frame head tensors (all length ``T``).

    Heads (each a 1×1 conv over the temporal embedding):
      * ``pitch``         (B, 3, T)   continuous, per voice
      * ``volume``        (B, 3, T)   continuous, per voice
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
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.in_proj = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden, dilation=d, kernel_size=kernel_size) for d in dilations
        )
        # Per-frame heads.
        self.head_pitch = nn.Conv1d(hidden, N_VOICES, 1)
        self.head_volume = nn.Conv1d(hidden, N_VOICES, 1)
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
        return {
            "pitch": self.head_pitch(h),
            "volume": self.head_volume(h),
            "tone_logit": self.head_tone(h),
            "noise_logit": self.head_noise(h),
            "env_use_logit": self.head_env_use(h),
            "noise_pitch": self.head_noise_pitch(h),
            "env_rate": self.head_env_rate(h),
            "env_shape": self.head_env_shape(h),
            "env_retrig": self.head_env_retrig(h),
        }
