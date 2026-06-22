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
from audio2ay4.train.warmstart import collate, pair_to_sample, train_step  # noqa: E402
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
    assert t["pitch"].shape == (3, 20)
    assert t["env_shape"].shape == (20,)
    assert t["env_shape"].dtype == np.int64
    assert 0 <= t["env_shape"].min() and t["env_shape"].max() < 16
    assert (t["env_rate"] > 0).all()  # log() in the loss needs strictly positive rates


def test_collate_pads_variable_lengths():
    samples = [pair_to_sample(_synthetic_pair(n, seed=n)) for n in (12, 18)]
    x, targets, pad = collate(samples)
    assert x.shape == (2, 32, 18)
    assert pad[0].sum().item() == 12 and pad[1].sum().item() == 18
    assert targets["pitch"].shape == (2, 3, 18)
    assert targets["env_shape"].shape == (2, 18)


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
