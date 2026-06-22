"""End-to-end pipeline tests using the dummy core.

The feature→core→compile path is exercised with only numpy; the chip writer/emulator path is
guarded with ``importorskip('audio2ay3')`` so it runs only when the proven emulator is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio2ay4 import io
from audio2ay4.chip.legality import is_legal
from audio2ay4.config import RunConfig
from audio2ay4.features import extract
from audio2ay4.models import get_core
from audio2ay4.repr import compile_state


def _write_sine_wav(path: str, sr: int = 44_100, freq: float = 440.0, secs: float = 0.5) -> None:
    t = np.arange(int(sr * secs), dtype=np.float32) / sr
    pcm = 0.6 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    io.encode(path, pcm, sr)


def test_features_and_dummy_core_compile_legal(tmp_path):
    cfg = RunConfig(core="dummy")
    wav = tmp_path / "tone.wav"
    _write_sine_wav(str(wav))

    audio = io.decode(str(wav), target_sr=cfg.sample_rate, mono=True)
    feats = extract(audio, cfg)
    assert feats.n_frames > 0

    state = get_core("dummy", cfg).infer(feats, cfg)
    assert len(state) == feats.n_frames

    song = compile_state(state, cfg)
    assert song.regs.dtype == np.uint8
    assert is_legal(song.regs)


def test_wav_roundtrip_io(tmp_path):
    cfg = RunConfig(core="dummy")
    wav = tmp_path / "rt.wav"
    _write_sine_wav(str(wav), secs=0.2)
    audio = io.decode(str(wav), target_sr=cfg.sample_rate, mono=True)
    assert audio.sample_rate == cfg.sample_rate
    assert audio.pcm.ndim == 1
    assert audio.duration_s > 0.0


def test_convert_writes_ym_when_ay3_available(tmp_path):
    pytest.importorskip("audio2ay3")
    from audio2ay4.convert import convert_audio_to_ym

    cfg = RunConfig(core="dummy")
    wav = tmp_path / "in.wav"
    out = tmp_path / "out.ym"
    _write_sine_wav(str(wav), secs=0.3)

    song = convert_audio_to_ym(str(wav), str(out), cfg)
    assert out.exists()
    assert song.n_frames > 0
    assert is_legal(song.regs)
