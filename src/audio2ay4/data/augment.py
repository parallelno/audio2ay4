"""Suno-style audio augmentation for the Regime-2 reward phase (design A.5 / domain gap).

The warm-start and Regime-1 both train on *clean chip* audio, so the encoder never sees the
coloration of a real SUNO-generated chiptune clip (stereo fold-down, reverb, EQ tilt, lossy-codec
artifacts, loudness maximisation). This module degrades a clean mono render to *look* like that
real-world input, so the reward phase can teach the encoder to recover the underlying AY controls
despite the coloration — i.e. it bridges the chip-audio → real-audio gap named in the design's risk
table ("Domain gap (chip-audio → real audio): reward phase trains on real + augmented audio").

Pure numpy (no torch / no librosa): it runs on the rendered reference audio inside the training loop
and the result is fed back through the *same* numpy ``features.extract`` used at inference, so the
training input distribution matches what the deployed model will actually receive.

Everything is driven by an explicit ``numpy.random.Generator`` so a run is reproducible and tests
are deterministic. ``strength`` in ``[0, 1]`` scales every effect's intensity and probability; 0 is
an identity passthrough.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["augment_audio"]

_F32 = np.float32


def _rms(x: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def _fft_linear_convolve(x: NDArray[np.float64], ir: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear convolution via FFT, returning the causal head aligned to ``x`` (length preserved)."""
    n = x.shape[0] + ir.shape[0] - 1
    nfft = 1 << (int(n - 1).bit_length())          # next power of two ≥ n
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)
    return y[: x.shape[0]]


def _spectral_colour(x: NDArray[np.float64], sr: int, rng: np.random.Generator,
                     s: float) -> NDArray[np.float64]:
    """Random EQ: spectral tilt + gentle low/high roll-off + one peaking band (lossy-codec feel)."""
    n = x.shape[0]
    f = np.fft.rfftfreq(n, d=1.0 / sr)
    nyq = sr / 2.0
    gain = np.ones_like(f)

    # Broadband tilt (±6 dB·s across the band).
    tilt_db = rng.uniform(-6.0, 6.0) * s
    gain *= 10.0 ** (tilt_db * (f / nyq - 0.5) / 20.0)

    # High roll-off — emulates lossy-codec low-pass; cutoff drops as strength rises.
    hc = rng.uniform(0.45 - 0.25 * s, 0.45) * nyq
    gain *= 1.0 / np.sqrt(1.0 + (f / max(hc, 1.0)) ** 6)

    # Gentle low cut (DC / rumble removal).
    lc = rng.uniform(20.0, 20.0 + 100.0 * s)
    r = f / max(lc, 1.0)
    gain *= r / np.sqrt(1.0 + r * r)

    # One peaking band.
    fc = rng.uniform(200.0, min(6000.0, nyq * 0.9))
    bw = fc * rng.uniform(0.3, 1.0)
    peak_db = rng.uniform(-6.0, 6.0) * s
    gain *= 1.0 + (10.0 ** (peak_db / 20.0) - 1.0) * np.exp(-0.5 * ((f - fc) / max(bw, 1.0)) ** 2)

    return np.fft.irfft(np.fft.rfft(x) * gain, n)


def _reverb(x: NDArray[np.float64], sr: int, rng: np.random.Generator,
            s: float) -> NDArray[np.float64]:
    """Short exponentially-decaying-noise reverb, mixed wet/dry."""
    dur = rng.uniform(0.03, 0.03 + 0.20 * s)
    m = max(4, int(dur * sr))
    if m >= x.shape[0]:
        return x
    t = np.arange(m)
    ir = rng.standard_normal(m) * np.exp(-t / (m / 4.0))
    ir[0] = 1.0                                     # keep the direct path
    ir /= np.sqrt(np.sum(ir * ir))
    wet = _fft_linear_convolve(x, ir)
    mix = rng.uniform(0.0, 0.5 * s)
    return (1.0 - mix) * x + mix * wet


def _companding_quantize(x: NDArray[np.float64], rng: np.random.Generator,
                         s: float) -> NDArray[np.float64]:
    """µ-law compand + coarse quantisation → lossy-codec-style quantisation noise."""
    peak = float(np.max(np.abs(x))) + 1e-9
    xn = x / peak
    mu = 255.0
    comp = np.sign(xn) * np.log1p(mu * np.abs(xn)) / np.log1p(mu)
    bits = rng.integers(6, 11)                      # 6..10 bits; lower = harsher
    levels = float(2 ** int(bits))
    comp = np.round(comp * levels) / levels
    expanded = np.sign(comp) * (1.0 / mu) * (np.expm1(np.abs(comp) * np.log1p(mu)))
    return (expanded * peak) * s + x * (1.0 - s)    # blend toward clean at low strength


def _dynamics(x: NDArray[np.float64], rng: np.random.Generator, s: float) -> NDArray[np.float64]:
    """Loudness push + tanh soft-limiting (loudness-maximisation feel)."""
    drive = 1.0 + rng.uniform(0.0, 4.0) * s
    k = rng.uniform(1.0, 1.0 + 3.0 * s)
    y = np.tanh(k * drive * x) / np.tanh(k)
    return y


def augment_audio(
    pcm: NDArray[np.floating],
    sample_rate: int,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
) -> NDArray[np.float32]:
    """Degrade a clean mono render to resemble a real SUNO chiptune clip.

    Applies (each gated by probability ∝ ``strength``): EQ colour, short reverb, µ-law companding
    quantisation, and dynamics/soft-limiting; then matches the output RMS back near the input level
    (× a small random factor) so the downstream log-mel sees realistic-but-bounded loudness.

    Returns a float32 mono array the same length as ``pcm``. ``strength == 0`` is identity.
    """
    x = np.asarray(pcm, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0 or x.shape[0] < 16:
        return x.astype(_F32)

    in_rms = _rms(x)
    if rng.random() < 0.9:
        x = _spectral_colour(x, sample_rate, rng, s)
    if rng.random() < 0.6:
        x = _reverb(x, sample_rate, rng, s)
    if rng.random() < 0.7:
        x = _companding_quantize(x, rng, s)
    if rng.random() < 0.7:
        x = _dynamics(x, rng, s)

    # Restore a realistic (bounded-random) loudness so the encoder stays level-robust.
    x *= (in_rms / _rms(x)) * rng.uniform(0.7, 1.3)
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x /= peak
    return x.astype(_F32)
