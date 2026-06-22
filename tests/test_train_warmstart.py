"""Warm-start training tests (CPU-only, no emulator render).

Synthetic pairs (random registers + random features) exercise the full target→collate→loss→step
path without touching audio2ay3, so they are safe on any machine. The overfit-one-batch test is the
§A.11 canary that the training loop is wired correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audio2ay4.config import DEFAULT_MASTER_CLOCK_HZ  # noqa: E402
from audio2ay4.data.pairing import TrainingPair  # noqa: E402
from audio2ay4.models.policy.network import ReversePlayer  # noqa: E402
from audio2ay4.repr.state import FeatureFrames  # noqa: E402
from audio2ay4.train.targets import build_targets  # noqa: E402
from audio2ay4.train.warmstart import _crop, _lr_at, collate, pair_to_sample, train_step  # noqa: E402
from audio2ay4.train.warmstart_loss import warmstart_loss  # noqa: E402


def _synthetic_pair(n_frames: int, dim: int = 32, seed: int = 0) -> TrainingPair:
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_frames, dim)).astype(np.float32)
    regs = rng.integers(0, 256, size=(n_frames, 16), dtype=np.uint8)
    return TrainingPair(
        feats=FeatureFrames(feats=feats, frame_rate=50, feat_kind="mel"),
        target_regs=regs,
        master_clock_hz=DEFAULT_MASTER_CLOCK_HZ,
        frame_rate_hz=50,
        meta={},
    )


def test_build_targets_shapes():
    rng = np.random.default_rng(1)
    regs = rng.integers(0, 256, size=(20, 16), dtype=np.uint8)
    t = build_targets(regs, DEFAULT_MASTER_CLOCK_HZ, 50)
    assert t["pitch_bin"].shape == (3, 20)
    assert t["pitch_bin"].dtype == np.int64
    assert 0 <= t["pitch_bin"].min() and t["pitch_bin"].max() < 61
    assert t["env_shape"].shape == (20,)
    assert t["env_shape"].dtype == np.int64
    assert 0 <= t["env_shape"].min() and t["env_shape"].max() < 16
    assert (t["env_rate"] > 0).all()  # log() in the loss needs strictly positive rates


def test_collate_pads_variable_lengths():
    samples = [pair_to_sample(_synthetic_pair(n, seed=n)) for n in (12, 18)]
    x, targets, pad = collate(samples)
    assert x.shape == (2, 32, 18)
    assert pad[0].sum().item() == 12 and pad[1].sum().item() == 18
    assert targets["pitch_bin"].shape == (2, 3, 18)
    assert targets["pitch_bin"].dtype == torch.int64
    assert targets["env_shape"].shape == (2, 18)


def test_crop_windows_long_and_keeps_short():
    rng = np.random.default_rng(0)
    long = pair_to_sample(_synthetic_pair(40, seed=1))
    feats, targets = _crop(long, 16, rng)
    assert feats.shape == (16, 32)
    assert targets["pitch_bin"].shape == (3, 16)
    assert targets["env_shape"].shape == (16,)
    short = pair_to_sample(_synthetic_pair(10, seed=2))
    assert _crop(short, 16, rng) is short  # shorter than window → unchanged


def test_lr_schedule_warmup_then_cosine_to_zero():
    base, steps, warmup = 1e-3, 1000, 100
    assert _lr_at(0, base, steps, warmup) == 0.0          # ramps from zero
    assert _lr_at(warmup, base, steps, warmup) == pytest.approx(base)  # peak at warmup end
    assert _lr_at(steps, base, steps, warmup) == pytest.approx(0.0, abs=1e-12)  # decays to ~0
    assert 0.0 < _lr_at(550, base, steps, warmup) < base  # somewhere in between mid-run


def test_warmstart_loss_is_finite():
    batch = collate([pair_to_sample(_synthetic_pair(16, seed=3))])
    net = ReversePlayer(in_dim=32, hidden=16).eval()
    with torch.no_grad():
        total, parts = warmstart_loss(net(batch[0]), batch[1], batch[2])
    assert np.isfinite(total.item())
    assert set(parts) == {
        "pitch", "volume", "tone", "noise", "env_use",
        "noise_pitch", "env_rate", "env_shape", "env_retrig",
    }


def test_overfit_one_batch_reduces_loss():
    torch.manual_seed(0)
    batch = collate([pair_to_sample(_synthetic_pair(16, seed=7))])
    net = ReversePlayer(in_dim=32, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    first, _ = train_step(net, opt, batch)
    last = first
    for _ in range(200):
        last, _ = train_step(net, opt, batch)
    assert last < 0.5 * first  # the loop can drive a single batch down
