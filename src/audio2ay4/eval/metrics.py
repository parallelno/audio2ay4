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
