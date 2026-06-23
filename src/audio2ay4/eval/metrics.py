"""Numpy metrics: perceptual spectral distance, register stability, legality rate."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..chip.legality import is_legal

_DEFAULT_FFTS = (512, 1024, 2048)


def _as_mono(x: NDArray) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    return a.mean(axis=1) if a.ndim > 1 else a


def _stft_logmag(x: NDArray[np.float64], n_fft: int, hop: int) -> NDArray[np.float64]:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    return np.log1p(np.abs(np.fft.rfft(frames * win, axis=1)))


def spectral_distance(a: NDArray, b: NDArray, ffts: tuple[int, ...] = _DEFAULT_FFTS) -> float:
    """Multi-resolution log-magnitude STFT L1 distance between two mono signals (0 = identical)."""
    xa, xb = _as_mono(a), _as_mono(b)
    n = min(len(xa), len(xb))
    if n == 0:
        return float("inf")
    xa, xb = xa[:n], xb[:n]
    total = 0.0
    for n_fft in ffts:
        hop = max(1, n_fft // 4)
        sa = _stft_logmag(xa, n_fft, hop)
        sb = _stft_logmag(xb, n_fft, hop)
        m = min(sa.shape[0], sb.shape[0])
        total += float(np.mean(np.abs(sa[:m] - sb[:m])))
    return total / len(ffts)


_A4_HZ = 440.0


def _stft_mag(x: NDArray[np.float64], n_fft: int, hop: int) -> NDArray[np.float64]:
    """Linear-magnitude STFT, frames along axis 0, rfft bins along axis 1."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * win, axis=1))


def _chroma_filterbank(
    n_fft: int, sr: int, fmin: float, fmax: float
) -> NDArray[np.float64]:
    """(12, n_bins) matrix folding each in-range rfft bin onto its pitch class."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((12, freqs.shape[0]), dtype=np.float64)
    for i, f in enumerate(freqs):
        if f < fmin or f > fmax:
            continue
        midi = 69.0 + 12.0 * np.log2(f / _A4_HZ)
        pc = int(np.round(midi)) % 12
        fb[pc, i] = 1.0
    return fb


def chroma_similarity(
    a: NDArray,
    b: NDArray,
    sr: int,
    *,
    n_fft: int = 4096,
    hop: int = 1024,
    fmin: float = 65.0,
    fmax: float = 2093.0,
) -> float:
    """Mean per-frame cosine of chroma (pitch-class) vectors in [0, 1] (HIGHER = better).

    Timbre-invariant: folds spectral energy onto the 12 pitch classes, so it rewards playing the
    *right notes/harmony* regardless of the chip's very different timbre. Frames silent in either
    signal are ignored.
    """
    xa, xb = _as_mono(a), _as_mono(b)
    n = min(len(xa), len(xb))
    if n < n_fft:
        return 0.0
    xa, xb = xa[:n], xb[:n]
    fb = _chroma_filterbank(n_fft, sr, fmin, fmax)
    ca = _stft_mag(xa, n_fft, hop) @ fb.T
    cb = _stft_mag(xb, n_fft, hop) @ fb.T
    m = min(ca.shape[0], cb.shape[0])
    if m == 0:
        return 0.0
    ca, cb = ca[:m], cb[:m]
    na = np.linalg.norm(ca, axis=1)
    nb = np.linalg.norm(cb, axis=1)
    valid = (na > 1e-6) & (nb > 1e-6)
    if not valid.any():
        return 0.0
    cos = np.sum(ca[valid] * cb[valid], axis=1) / (na[valid] * nb[valid])
    return float(np.mean(cos))


def onset_similarity(
    a: NDArray, b: NDArray, sr: int, *, n_fft: int = 2048, hop: int = 512
) -> float:
    """Pearson correlation of spectral-flux onset envelopes in [-1, 1] (HIGHER = better).

    Captures rhythm/timing adherence (e.g. whether the noise channel's drums land with the
    original), independent of timbre and loudness.
    """
    xa, xb = _as_mono(a), _as_mono(b)
    n = min(len(xa), len(xb))
    if n < 2 * n_fft:
        return 0.0

    def _onset_env(x: NDArray[np.float64]) -> NDArray[np.float64]:
        mag = _stft_mag(x, n_fft, hop)
        flux = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
        return flux

    ea, eb = _onset_env(xa[:n]), _onset_env(xb[:n])
    m = min(ea.shape[0], eb.shape[0])
    if m < 2:
        return 0.0
    ea, eb = ea[:m] - ea[:m].mean(), eb[:m] - eb[:m].mean()
    da, db = float(np.linalg.norm(ea)), float(np.linalg.norm(eb))
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return float(np.dot(ea, eb) / (da * db))


def stability(regs: NDArray[np.uint8]) -> float:
    """Frame-to-frame change rate in [0, 1] (lower = steadier; flags period/volume thrash)."""
    r = np.asarray(regs)
    if r.ndim != 2 or r.shape[0] < 2:
        return 0.0
    return float(np.mean(np.any(r[1:] != r[:-1], axis=1)))


def legality_rate(reg_streams: list[NDArray[np.uint8]]) -> float:
    """Fraction of register streams that are fully hardware-legal (target: 1.0)."""
    if not reg_streams:
        return 1.0
    return float(np.mean([1.0 if is_legal(r) else 0.0 for r in reg_streams]))
