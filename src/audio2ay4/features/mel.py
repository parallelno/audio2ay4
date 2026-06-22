"""Numpy log-mel spectrogram, frame-aligned to the AY frame grid (no torch/librosa needed)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import RunConfig
from ..repr.state import AudioBuffer, FeatureFrames

_N_MELS = 80
_N_FFT = 2048
_FMIN = 30.0


def _hz_to_mel(f: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: NDArray[np.float64]) -> NDArray[np.float64]:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> NDArray[np.float64]:
    mel_pts = np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


def mel_center_freqs(n_mels: int = _N_MELS, n_fft: int = _N_FFT,
                     sr: int = 44_100, fmin: float = _FMIN, fmax: float | None = None) -> NDArray[np.float64]:
    """Approximate centre frequency (Hz) of each mel band — used by the dummy core's pitch guess."""
    fmax = fmax or sr / 2.0
    mel_pts = np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    return _mel_to_hz(mel_pts)[1:-1]


def extract(audio: AudioBuffer, cfg: RunConfig, n_mels: int = _N_MELS,
            n_fft: int = _N_FFT) -> FeatureFrames:
    """Compute a log-mel spectrogram with one column per AY frame (hop = sr / frame_rate)."""
    pcm = np.asarray(audio.pcm, dtype=np.float64)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    sr = int(audio.sample_rate)
    hop = max(1, round(sr / cfg.frame_rate_hz))
    n_frames = max(1, int(np.ceil(len(pcm) / hop)))

    window = np.hanning(n_fft)
    half = n_fft // 2
    fb = _mel_filterbank(n_mels, n_fft, sr, _FMIN, sr / 2.0)
    padded = np.pad(pcm, (half, half + n_frames * hop), mode="constant")

    feats = np.empty((n_frames, n_mels), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        seg = padded[start:start + n_fft] * window
        mag = np.abs(np.fft.rfft(seg))
        feats[i] = np.log1p(fb @ mag).astype(np.float32)

    return FeatureFrames(feats=feats, frame_rate=int(cfg.frame_rate_hz), feat_kind="mel")
