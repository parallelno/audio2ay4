"""Tests for the pseudo-CQT feature front-end and the feat_kind dispatch."""

from __future__ import annotations

import numpy as np

from audio2ay4.config import RunConfig
from audio2ay4.features import extract
from audio2ay4.features.cqt import cqt_center_freqs
from audio2ay4.features.cqt import extract as cqt_extract
from audio2ay4.repr.state import AudioBuffer


def _tone(freq: float, sr: int = 44_100, dur: float = 1.0) -> AudioBuffer:
    t = np.arange(int(sr * dur)) / sr
    pcm = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return AudioBuffer(pcm=pcm, sample_rate=sr, duration_s=dur)


def test_cqt_shape_and_kind():
    cfg = RunConfig(feat_kind="cqt", frame_rate_hz=50, sample_rate=44_100)
    ff = cqt_extract(_tone(440.0), cfg)
    assert ff.feat_kind == "cqt"
    assert ff.feats.ndim == 2
    assert ff.feats.shape[1] == 12 * 7        # bins_per_octave * n_octaves
    assert ff.frame_rate == 50


def test_cqt_peak_tracks_pitch():
    cfg = RunConfig(feat_kind="cqt", frame_rate_hz=50, sample_rate=44_100)
    centers = cqt_center_freqs()
    for freq in (220.0, 440.0, 880.0):
        ff = cqt_extract(_tone(freq), cfg)
        mean_bin = ff.feats.mean(axis=0)
        peak = int(np.argmax(mean_bin))
        # nearest semitone bin to the played frequency
        expected = int(np.argmin(np.abs(centers - freq)))
        assert abs(peak - expected) <= 1, f"{freq}Hz → bin {peak}, expected ~{expected}"


def test_extract_dispatches_on_feat_kind():
    mel_cfg = RunConfig(feat_kind="mel", frame_rate_hz=50, sample_rate=44_100)
    cqt_cfg = RunConfig(feat_kind="cqt", frame_rate_hz=50, sample_rate=44_100)
    audio = _tone(330.0)
    assert extract(audio, mel_cfg).feat_kind == "mel"
    assert extract(audio, cqt_cfg).feat_kind == "cqt"
