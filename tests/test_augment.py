"""A5 — SUNO-style audio augmentation (domain-gap bridge). Pure numpy, no torch needed."""

from __future__ import annotations

import numpy as np

from audio2ay4.data.augment import augment_audio


def _tone(sr: int = 8_000, hz: float = 220.0, secs: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_strength_zero_is_identity():
    x = _tone()
    out = augment_audio(x, 8_000, np.random.default_rng(0), strength=0.0)
    assert out.dtype == np.float32
    assert np.allclose(out, x)


def test_shape_dtype_and_bounds_preserved():
    x = _tone()
    out = augment_audio(x, 8_000, np.random.default_rng(1), strength=1.0)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0 + 1e-5


def test_deterministic_given_seed():
    x = _tone()
    a = augment_audio(x, 8_000, np.random.default_rng(7), strength=1.0)
    b = augment_audio(x, 8_000, np.random.default_rng(7), strength=1.0)
    assert np.array_equal(a, b)


def test_actually_changes_audio():
    x = _tone()
    out = augment_audio(x, 8_000, np.random.default_rng(3), strength=1.0)
    # Some real coloration must have happened (not a near-identity passthrough).
    assert not np.allclose(out, x, atol=1e-3)


def test_handles_tiny_input():
    x = np.zeros(8, np.float32)
    out = augment_audio(x, 8_000, np.random.default_rng(0), strength=1.0)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_stereo_is_folded_to_mono():
    x = _tone()
    stereo = np.stack([x, x], axis=1)
    out = augment_audio(stereo, 8_000, np.random.default_rng(2), strength=1.0)
    assert out.ndim == 1
    assert out.shape[0] == x.shape[0]


def test_silence_stays_finite():
    x = np.zeros(4_000, np.float32)
    out = augment_audio(x, 8_000, np.random.default_rng(5), strength=1.0)
    assert np.all(np.isfinite(out))
