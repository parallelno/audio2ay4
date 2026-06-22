"""Pure tests for the deterministic register compiler — no audio2ay3 required."""

from __future__ import annotations

import math

import numpy as np

from audio2ay4.chip.legality import is_legal
from audio2ay4.config import RunConfig
from audio2ay4.repr import compile_state, parse_song
from audio2ay4.repr.state import (
    AYGlobalFrame,
    AYStateFrame,
    AYVoiceFrame,
    silent_state,
)


def _cfg() -> RunConfig:
    return RunConfig(core="dummy")


def _varied_state():
    frames = []
    for i in range(20):
        a = AYVoiceFrame(pitch_semitones=48 + i, volume_db=-3.0 * (i % 5), tone_on=True)
        b = AYVoiceFrame(pitch_semitones=60.0, volume_db=0.0, tone_on=True, noise_on=True)
        c = AYVoiceFrame(pitch_semitones=72.0, volume_db=-12.0, tone_on=False, use_envelope=True)
        glob = AYGlobalFrame(noise_pitch=i / 20.0, env_shape=i % 16, env_rate=2.0 + i,
                             env_retrigger=(i % 4 == 0))
        frames.append(AYStateFrame(voices=(a, b, c), glob=glob))
    return frames


def test_compile_shape_dtype_and_legal():
    song = compile_state(_varied_state(), _cfg())
    assert song.regs.shape == (20, 16)
    assert song.regs.dtype == np.uint8
    assert is_legal(song.regs)


def test_silent_state_is_legal():
    song = compile_state(silent_state(10), _cfg())
    assert is_legal(song.regs)
    # all tone+noise disabled ⇒ mixer R7 low 6 bits set
    assert np.all(song.regs[:, 7] & 0x3F == 0x3F)


def test_octave_folding_keeps_period_in_range():
    cfg = _cfg()
    # absurdly low and high pitches must still produce a legal 12-bit period
    low = AYVoiceFrame(pitch_semitones=-40.0, volume_db=0.0, tone_on=True)
    high = AYVoiceFrame(pitch_semitones=140.0, volume_db=0.0, tone_on=True)
    state = [AYStateFrame(voices=(low, high, low), glob=AYGlobalFrame())]
    song = compile_state(state, cfg)
    assert is_legal(song.regs)
    for c in range(3):
        tp = ((int(song.regs[0, 2 * c + 1]) & 0x0F) << 8) | int(song.regs[0, 2 * c])
        assert 1 <= tp <= 4095


def test_env_no_write_sentinel_when_not_retriggered():
    v = AYVoiceFrame(pitch_semitones=60.0, volume_db=0.0, tone_on=True)
    glob = AYGlobalFrame(env_shape=5, env_retrigger=False)
    song = compile_state([AYStateFrame(voices=(v, v, v), glob=glob)], _cfg())
    assert int(song.regs[0, 13]) == 0xFF


def test_parse_roundtrip_recovers_pitch():
    cfg = _cfg()
    v = AYVoiceFrame(pitch_semitones=57.0, volume_db=0.0, tone_on=True)
    silent = AYVoiceFrame(pitch_semitones=float("nan"), volume_db=float("-inf"), tone_on=False)
    song = compile_state([AYStateFrame(voices=(v, silent, silent), glob=AYGlobalFrame())], cfg)
    decoded = parse_song(song)
    a = decoded[0].voices[0]
    assert a.tone_on
    assert math.isfinite(a.pitch_semitones)
    assert abs(a.pitch_semitones - 57.0) < 1.0  # quantisation tolerance
