"""Pure tests for eval metrics."""

from __future__ import annotations

import numpy as np

from audio2ay4.eval import legality_rate, spectral_distance, stability


def _sine(freq: float, sr: int = 44_100, secs: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_spectral_distance_zero_for_identical():
    x = _sine(440.0)
    assert spectral_distance(x, x) < 1e-6


def test_spectral_distance_positive_for_different():
    a, b = _sine(220.0), _sine(880.0)
    assert spectral_distance(a, b) > spectral_distance(a, a)


def test_spectral_distance_handles_length_mismatch():
    a, b = _sine(440.0, secs=0.5), _sine(440.0, secs=0.3)
    d = spectral_distance(a, b)
    assert np.isfinite(d) and d >= 0.0


def test_stability_zero_for_constant_then_positive():
    const = np.tile(np.arange(16, dtype=np.uint8), (10, 1))
    assert stability(const) == 0.0
    changing = const.copy()
    changing[5, 0] = 99
    assert stability(changing) > 0.0


def test_legality_rate():
    legal = np.zeros((4, 16), dtype=np.uint8)
    illegal = np.zeros((4, 16), dtype=np.uint8)
    illegal[0, 1] = 0xF0  # tone-high nibble set ⇒ illegal
    assert legality_rate([legal, legal]) == 1.0
    assert legality_rate([legal, illegal]) == 0.5
    assert legality_rate([]) == 1.0
