"""Tests for the real-audio analysis-by-synthesis harness (``train.real_train``).

Pure-logic parts (prompt-stem grouping, no-leak split) run anywhere. The forward-pass test builds a
tiny ``ReversePlayer`` + ``DiffAyEmulator`` on synthetic features/PCM and checks the chroma/onset
reward is finite and differentiable — no audio2ay3 and no real clips needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio2ay4.train.real_train import prompt_stem, split_by_stem

torch = pytest.importorskip("torch")

from audio2ay4.chip.diff import DiffAyEmulator  # noqa: E402
from audio2ay4.config import DEFAULT_MASTER_CLOCK_HZ  # noqa: E402
from audio2ay4.models.policy.network import ReversePlayer  # noqa: E402
from audio2ay4.train.real_train import RealSample, real_forward  # noqa: E402
from audio2ay4.train.reward import RewardWeights  # noqa: E402


def test_prompt_stem_groups_variants():
    assert prompt_stem("Pixel Drift (3).mp3") == "pixel drift"
    assert prompt_stem("Pixel_Drift.mp3") == "pixel drift"
    assert prompt_stem("pixel drift.mp3") == "pixel drift"
    assert prompt_stem("The_Last_Pixel.mp3") == prompt_stem("The Last Pixel (1).mp3")


def test_split_by_stem_no_leak():
    paths = [
        "a/Pixel Drift.mp3", "a/Pixel Drift (1).mp3", "a/Pixel Drift (2).mp3",
        "a/Goblins_Lair.mp3", "a/Start Button.mp3", "a/Start Button (1).mp3",
        "a/The Last Pixel.mp3", "a/Coin Drop.mp3",
    ]
    train, heldout = split_by_stem(paths, heldout_frac=0.3, seed=0)
    # partition: every clip lands on exactly one side
    assert sorted(train + heldout) == sorted(paths)
    assert set(train).isdisjoint(heldout)
    # no prompt-stem straddles the split (the whole point: no near-duplicate leakage)
    train_stems = {prompt_stem(p) for p in train}
    heldout_stems = {prompt_stem(p) for p in heldout}
    assert train_stems.isdisjoint(heldout_stems)
    # training is never starved of all clips
    assert train


def test_split_keeps_training_clips_when_frac_high():
    paths = ["a/One.mp3", "a/Two.mp3", "a/Three.mp3"]
    train, _heldout = split_by_stem(paths, heldout_frac=1.0, seed=1)
    assert train  # at least one stem stays for training


def _real_sample(stem: str, n_frames: int, dim: int, seed: int) -> RealSample:
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_frames, dim)).astype(np.float32)
    # PCM long enough for the chroma n_fft (4096) and onset 2*2048.
    pcm = (0.1 * rng.standard_normal(n_frames * 882)).astype(np.float32)
    return stem, feats, pcm, DEFAULT_MASTER_CLOCK_HZ, 50


def test_real_forward_finite_and_differentiable():
    dim = 32
    net = ReversePlayer(in_dim=dim, hidden=16)
    emu = DiffAyEmulator(render_sr=44_100, oversample=1, max_partials=8)
    samples = [_real_sample("a", 20, dim, 0), _real_sample("b", 24, dim, 1)]
    weights = RewardWeights(spectral=0.0, jitter=0.0, chroma=5.0, onset=1.0)
    total, parts = real_forward(net, emu, samples, device="cpu", weights=weights, tau=1.0)
    assert torch.isfinite(total)
    assert "chroma" in parts and "onset" in parts
    total.backward()
    grad = next(p.grad for p in net.parameters() if p.grad is not None)
    assert torch.isfinite(grad).all()
