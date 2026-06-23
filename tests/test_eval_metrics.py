"""Pure tests for eval metrics."""

from __future__ import annotations

import numpy as np

from audio2ay4.eval import (
    chroma_similarity,
    legality_rate,
    onset_similarity,
    spectral_distance,
    stability,
)


def _sine(freq: float, sr: int = 44_100, secs: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _square(freq: float, sr: int = 44_100, secs: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * secs)) / sr
    return (0.4 * np.sign(np.sin(2 * np.pi * freq * t))).astype(np.float32)


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


def test_chroma_similarity_one_for_identical():
    x = _square(220.0)
    assert chroma_similarity(x, x, 44_100) > 0.99


def test_chroma_similarity_timbre_invariant_same_note():
    # Same pitch class, very different timbre (sine vs square) ⇒ still high chroma agreement.
    sine, square = _sine(220.0), _square(220.0)
    same = chroma_similarity(sine, square, 44_100)
    # A tritone away (different notes) should score clearly lower than the same note.
    tritone = chroma_similarity(sine, _square(220.0 * 2 ** (6 / 12)), 44_100)
    assert same > 0.8
    assert same > tritone


def test_chroma_similarity_octave_equivalence():
    # An octave up is the same pitch class ⇒ high agreement despite different frequency.
    base, octave = _square(220.0), _square(440.0)
    assert chroma_similarity(base, octave, 44_100) > 0.8


def test_chroma_similarity_handles_short_or_silent():
    assert chroma_similarity(np.zeros(10, np.float32), np.zeros(10, np.float32), 44_100) == 0.0
    silent = np.zeros(44_100, np.float32)
    assert chroma_similarity(silent, silent, 44_100) == 0.0


def test_onset_similarity_one_for_identical_rhythm():
    sr = 44_100
    # A pulse train: short bursts separated by silence.
    sig = np.zeros(sr, np.float32)
    burst = sr // 32
    for start in range(0, sr - burst, sr // 8):
        sig[start:start + burst] = 0.5
    assert onset_similarity(sig, sig, sr) > 0.99


def test_onset_similarity_lower_for_mismatched_rhythm():
    sr = 44_100

    def pulses(period_frac: int) -> np.ndarray:
        s = np.zeros(sr, np.float32)
        burst = sr // 64
        for start in range(0, sr - burst, sr // period_frac):
            s[start:start + burst] = 0.5
        return s

    a = pulses(8)
    same = onset_similarity(a, a, sr)
    diff = onset_similarity(a, pulses(5), sr)
    assert same > diff


def test_onset_similarity_handles_silence():
    silent = np.zeros(44_100, np.float32)
    assert onset_similarity(silent, silent, 44_100) == 0.0

