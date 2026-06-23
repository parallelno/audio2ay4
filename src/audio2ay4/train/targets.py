"""Supervised warm-start targets: registers → per-head target arrays (numpy, torch-free).

Inverts a ground-truth register stream into the same quantities the reverse player's heads emit,
so the warm-start loss can regress/classify head-by-head (design A.5). Uses ``parse_song`` as the
single canonical registers→AYState inverse, then packs frames into arrays. Continuous targets are
clamped to the heads' representable ranges (``spec``) so the loss is achievable.
"""

from __future__ import annotations

import math

import numpy as np

from ..models.policy.spec import (
    ENV_RATE_FLOOR_HZ,
    N_VOICES,
    PITCH_CENTER,
    VOL_FLOOR_DB,
    db_to_vol_level,
    env_rate_to_bin,
    noise_pitch_to_level,
    pitch_to_bin,
)
from ..repr import parse_song
from ..repr.state import YmSong


def build_targets(regs: np.ndarray, master_clock_hz: int, frame_rate_hz: int) -> dict[str, np.ndarray]:
    """Pack ground-truth ``regs`` (T, 16) into per-head target arrays.

    Returns a dict (T = n_frames):
      ``pitch_bin`` (3, T) int64 (semitone bin index), ``volume_level`` (3, T) int64 (DAC level),
      ``tone`` / ``noise`` / ``env_use`` (3, T) f32{0,1},
      ``noise_pitch_level`` (T,) int64 (noise-period level), ``env_rate_bin`` (T,) int64,
      ``env_shape`` (T,) int64, ``env_retrig`` (T,) f32{0,1}.
      Masks are derived by the loss from the gate targets.
    """
    state = parse_song(YmSong(regs=np.asarray(regs, np.uint8),
                              master_clock_hz=master_clock_hz, frame_rate_hz=frame_rate_hz))
    t_len = len(state)
    pitch = np.full((N_VOICES, t_len), pitch_to_bin(PITCH_CENTER), np.int64)
    volume = np.full((N_VOICES, t_len), db_to_vol_level(VOL_FLOOR_DB), np.int64)
    tone = np.zeros((N_VOICES, t_len), np.float32)
    noise = np.zeros((N_VOICES, t_len), np.float32)
    env_use = np.zeros((N_VOICES, t_len), np.float32)
    noise_pitch = np.zeros(t_len, np.int64)  # noise-period class index
    env_rate = np.zeros(t_len, np.int64)  # log-spaced rate bin; floor ⇒ bin 0
    env_shape = np.zeros(t_len, np.int64)
    env_retrig = np.zeros(t_len, np.float32)

    for t, frame in enumerate(state):
        for c, v in enumerate(frame.voices):
            tone[c, t] = 1.0 if v.tone_on else 0.0
            noise[c, t] = 1.0 if v.noise_on else 0.0
            env_use[c, t] = 1.0 if v.use_envelope else 0.0
            if math.isfinite(v.pitch_semitones):
                pitch[c, t] = pitch_to_bin(v.pitch_semitones)
            volume[c, t] = db_to_vol_level(v.volume_db)
        g = frame.glob
        noise_pitch[t] = noise_pitch_to_level(float(np.clip(g.noise_pitch, 0.0, 1.0)))
        env_rate[t] = env_rate_to_bin(max(g.env_rate, ENV_RATE_FLOOR_HZ))
        env_shape[t] = int(g.env_shape) & 0x0F
        env_retrig[t] = 1.0 if g.env_retrigger else 0.0

    return {
        "pitch_bin": pitch, "volume_level": volume, "tone": tone, "noise": noise,
        "env_use": env_use, "noise_pitch_level": noise_pitch, "env_rate_bin": env_rate,
        "env_shape": env_shape, "env_retrig": env_retrig,
    }
