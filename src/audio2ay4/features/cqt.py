"""Numpy pseudo-CQT: a log-frequency, semitone-spaced spectrogram aligned to the AY frame grid.

Mel's linear-Hz triangular bands give poor low-frequency resolution — the likely ceiling on the
learned core's *pitch* accuracy (chroma). This front-end replaces them with a **constant-Q-style
log-frequency filterbank**: centres spaced ``bins_per_octave`` per octave (12 ⇒ one bin per
semitone, aligned with the pitch head and the chroma metric), over a **large FFT** so the low
octaves are actually resolved. Same ``FeatureFrames`` contract and framing (hop = sr / frame_rate)
as :mod:`mel`, torch-free, so it drops straight into ``features.extract`` behind ``feat_kind="cqt"``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import RunConfig
from ..repr.state import AudioBuffer, FeatureFrames

_FMIN = 32.70           # C1
_BINS_PER_OCTAVE = 12   # one bin per semitone (matches the chroma / pitch grid)
_N_OCTAVES = 7          # C1 → C8 (~4186 Hz)
_N_FFT = 8192           # long window: resolves the low octaves mel cannot


def _logf_filterbank(
    n_bins: int, bins_per_octave: int, n_fft: int, sr: int, fmin: float
) -> NDArray[np.float64]:
    """Triangular log-frequency (constant-Q-ish) filterbank → ``(n_bins, n_fft//2+1)``.

    Centres are geometric (``fmin·2^(k/bpo)``); each filter rises from the previous centre to its
    own and falls to the next — the log-spaced analogue of the mel triangles.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)                    # (n_fft//2+1,)
    centers = fmin * 2.0 ** (np.arange(-1, n_bins + 1) / bins_per_octave)  # k = -1 … n_bins
    fb = np.zeros((n_bins, freqs.shape[0]), dtype=np.float64)
    for k in range(1, n_bins + 1):
        left, center, right = centers[k - 1], centers[k], centers[k + 1]
        rise = (freqs >= left) & (freqs <= center)
        fb[k - 1, rise] = (freqs[rise] - left) / max(center - left, 1e-9)
        fall = (freqs > center) & (freqs <= right)
        fb[k - 1, fall] = (right - freqs[fall]) / max(right - center, 1e-9)
    return fb


def cqt_center_freqs(
    n_bins: int = _BINS_PER_OCTAVE * _N_OCTAVES,
    bins_per_octave: int = _BINS_PER_OCTAVE,
    fmin: float = _FMIN,
) -> NDArray[np.float64]:
    """Centre frequency (Hz) of each log-frequency bin."""
    return fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)


def extract(
    audio: AudioBuffer,
    cfg: RunConfig,
    *,
    bins_per_octave: int = _BINS_PER_OCTAVE,
    n_octaves: int = _N_OCTAVES,
    n_fft: int = _N_FFT,
) -> FeatureFrames:
    """Compute a log-frequency (pseudo-CQT) spectrogram, one column per AY frame."""
    pcm = np.asarray(audio.pcm, dtype=np.float64)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    sr = int(audio.sample_rate)
    n_bins = bins_per_octave * n_octaves
    hop = max(1, round(sr / cfg.frame_rate_hz))
    n_frames = max(1, int(np.ceil(len(pcm) / hop)))

    window = np.hanning(n_fft)
    half = n_fft // 2
    fb = _logf_filterbank(n_bins, bins_per_octave, n_fft, sr, _FMIN)
    padded = np.pad(pcm, (half, half + n_frames * hop), mode="constant")

    feats = np.empty((n_frames, n_bins), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        seg = padded[start:start + n_fft] * window
        mag = np.abs(np.fft.rfft(seg))
        feats[i] = np.log1p(fb @ mag).astype(np.float32)

    return FeatureFrames(feats=feats, frame_rate=int(cfg.frame_rate_hz), feat_kind="cqt")


__all__ = ["extract", "cqt_center_freqs"]
