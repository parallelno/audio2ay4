"""Plan A ``rl`` core smoke tests (CPU-only, no emulator/audio).

Validates the A1 canary: an **untrained** reverse player produces an ``AYState`` that the
deterministic compiler turns into a hardware-legal register stream, deterministically. Heavy
training and any emulator-rendering tests run on the second machine; these only need torch + numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audio2ay4.chip.legality import is_legal  # noqa: E402
from audio2ay4.config import RunConfig  # noqa: E402
from audio2ay4.models import get_core  # noqa: E402
from audio2ay4.repr import compile_state  # noqa: E402
from audio2ay4.repr.state import FeatureFrames  # noqa: E402


def _fake_feats(n_frames: int = 40, dim: int = 80, seed: int = 0) -> FeatureFrames:
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_frames, dim)).astype(np.float32)
    return FeatureFrames(feats=feats, frame_rate=50, feat_kind="mel")


def test_rl_core_infers_legal_state():
    cfg = RunConfig(core="rl")
    feats = _fake_feats(n_frames=40)
    state = get_core("rl", cfg).infer(feats, cfg)

    assert len(state) == feats.n_frames
    for frame in state:
        assert len(frame.voices) == 3

    song = compile_state(state, cfg)
    assert song.regs.dtype == np.uint8
    assert song.n_frames == feats.n_frames
    assert is_legal(song.regs)


def test_rl_core_is_deterministic():
    cfg = RunConfig(core="rl", seed=123)
    feats = _fake_feats(n_frames=24)
    a = compile_state(get_core("rl", cfg).infer(feats, cfg), cfg)
    b = compile_state(get_core("rl", cfg).infer(feats, cfg), cfg)
    assert np.array_equal(a.regs, b.regs)


def test_rl_core_handles_empty_input():
    cfg = RunConfig(core="rl")
    feats = FeatureFrames(feats=np.zeros((0, 80), np.float32), frame_rate=50, feat_kind="mel")
    assert get_core("rl", cfg).infer(feats, cfg) == []


def test_reverse_player_head_shapes():
    from audio2ay4.models.policy.network import ReversePlayer
    from audio2ay4.models.policy.spec import N_ENV_RATE_BINS, N_PITCH_BINS, N_VOL_LEVELS

    net = ReversePlayer(in_dim=80, hidden=32).eval()
    x = torch.randn(2, 80, 17)  # (B, in_dim, T)
    with torch.no_grad():
        out = net(x)
    assert out["pitch_logits"].shape == (2, 3, N_PITCH_BINS, 17)
    assert out["volume_logits"].shape == (2, 3, N_VOL_LEVELS, 17)
    assert out["env_rate_logits"].shape == (2, N_ENV_RATE_BINS, 17)
    expected = {
        "tone_logit": 3, "noise_logit": 3, "env_use_logit": 3,
        "noise_pitch": 1, "env_shape": 16, "env_retrig": 1,
    }
    for name, ch in expected.items():
        assert out[name].shape == (2, ch, 17), name
