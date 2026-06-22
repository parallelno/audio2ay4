"""Decode/encode implementations. Light core handles ``.wav`` via stdlib; other formats use the
optional ``soundfile`` backend."""

from __future__ import annotations

import os
import wave

import numpy as np
from numpy.typing import NDArray

from ..repr.state import AudioBuffer

_SF_HINT = (
    "Reading/writing this format needs the optional audio backend.\n"
    "Install it with:  pip install -e .[audio]   (soundfile/libsndfile)\n"
    "Or convert to/from .wav, which the light core supports natively."
)


def _try_soundfile():
    try:
        import soundfile  # noqa: F401

        return soundfile
    except Exception:
        return None


# --- read -----------------------------------------------------------------------------------

def _read_wav_stdlib(path: str) -> tuple[NDArray[np.float32], int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:  # unsigned 8-bit PCM
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:  # pragma: no cover
        raise ValueError(f"Unsupported WAV sample width: {width * 8} bit")
    if n_ch > 1:
        data = data.reshape(-1, n_ch)
    return data, sr


def _resample_linear(pcm: NDArray[np.float32], sr_in: int, sr_out: int) -> NDArray[np.float32]:
    if sr_in == sr_out or len(pcm) == 0:
        return pcm
    n_out = int(round(len(pcm) * sr_out / sr_in))
    if n_out <= 0:
        return pcm[:0]
    x_in = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    if pcm.ndim == 1:
        return np.interp(x_out, x_in, pcm).astype(np.float32)
    cols = [np.interp(x_out, x_in, pcm[:, c]) for c in range(pcm.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def decode(path: str, target_sr: int, mono: bool = True) -> AudioBuffer:
    """Decode ``path`` to float32 PCM at ``target_sr`` (mono by default)."""
    sf = _try_soundfile()
    ext = os.path.splitext(path)[1].lower()
    if sf is not None:
        pcm, sr = sf.read(path, dtype="float32", always_2d=False)
        pcm = np.asarray(pcm, dtype=np.float32)
    elif ext == ".wav":
        pcm, sr = _read_wav_stdlib(path)
    else:
        raise RuntimeError(f"Cannot decode '{ext}' in the light core. {_SF_HINT}")

    if mono and pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    pcm = _resample_linear(pcm, sr, target_sr)
    return AudioBuffer(pcm=pcm, sample_rate=int(target_sr), duration_s=len(pcm) / float(target_sr))


# --- write ----------------------------------------------------------------------------------

def _write_wav_stdlib(path: str, pcm: NDArray[np.float32], sr: int) -> None:
    data = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    if data.ndim == 1:
        n_ch = 1
        interleaved = data
    else:
        n_ch = data.shape[1]
        interleaved = data.reshape(-1)
    pcm16 = (interleaved * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_ch)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


def encode(path: str, pcm: NDArray[np.float32], sr: int) -> None:
    """Encode float32 PCM to ``path``. ``.wav`` always works; other formats need ``[audio]``."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        _write_wav_stdlib(path, pcm, sr)
        return
    sf = _try_soundfile()
    if sf is None:
        raise RuntimeError(f"Cannot encode '{ext}' in the light core. {_SF_HINT}")
    sf.write(path, np.asarray(pcm, dtype=np.float32), int(sr))
