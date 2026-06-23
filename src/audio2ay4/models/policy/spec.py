"""Shared head specification for the Plan A reverse player (torch-free).

Single source of truth for how the network's raw heads map to AYState units, imported by both the
torch side (``network``/``core``/training ``loss``) and the numpy side (training ``targets``). Kept
free of torch so the deterministic core and the target builder import without the neural extra.
"""

from __future__ import annotations

import math

from ...repr.compile import (
    NP_MAX,
    NP_MIN,
    db_to_level,
    level_to_db,
    noise_pitch_to_np,
    np_to_noise_pitch,
)

N_VOICES = 3
N_ENV_SHAPES = 16

# Decode ranges — bound raw heads to sane musical values (the compiler still clamps to hardware).
PITCH_CENTER = 60.0       # MIDI C4 (centre of the tanh-bounded pitch head)
PITCH_SPAN = 30.0         # ±30 semitones ⇒ ~C1½..F#6
VOL_FLOOR_DB = -60.0
VOL_CEIL_DB = 0.0
ENV_RATE_FLOOR_HZ = 0.1

# Pitch is predicted as a classification over a uniform semitone grid (not regressed): a raw MSE
# pitch head collapses to the corpus mean, so we quantise [PITCH_MIN, PITCH_MAX] into N_PITCH_BINS
# integer-semitone bins and learn a softmax over them. Decode picks the arg-max bin centre.
PITCH_MIN = PITCH_CENTER - PITCH_SPAN          # 30.0 semitones (~C1½)
PITCH_MAX = PITCH_CENTER + PITCH_SPAN          # 90.0 semitones (~F#6)
N_PITCH_BINS = 61                              # 1-semitone resolution over the 60-semitone span
PITCH_BIN_WIDTH = (PITCH_MAX - PITCH_MIN) / (N_PITCH_BINS - 1)  # semitones per bin


def pitch_to_bin(semitones: float) -> int:
    """Quantise a pitch (semitones) to its nearest classification bin index in ``[0, N_PITCH_BINS)``."""
    frac = (semitones - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)
    b = round(frac * (N_PITCH_BINS - 1))
    return int(min(max(b, 0), N_PITCH_BINS - 1))


def bin_to_pitch(b: int) -> float:
    """Inverse of :func:`pitch_to_bin`: bin index → semitones (bin centre)."""
    return PITCH_MIN + b * PITCH_BIN_WIDTH


# Volume is predicted as a classification over the 4-bit DAC's 16 amplitude levels (not regressed
# in dB): the AY DAC is quantised to 16 discrete steps, so regressing a continuous dB collapses to
# the corpus mean exactly like the raw pitch head did. The dB↔level mapping is the compiler's
# canonical amplitude table (single source of truth); decode picks the arg-max level → dB.
N_VOL_LEVELS = 16  # 4-bit AY DAC amplitude levels (0..15)


def db_to_vol_level(db: float) -> int:
    """Perceptual dB → nearest 4-bit DAC level index in ``[0, N_VOL_LEVELS)`` (compiler table)."""
    return db_to_level(db)


def vol_level_to_db(level: int) -> float:
    """Inverse: DAC level index → dB via the compiler's amplitude table (level 0 ⇒ ``-inf``)."""
    return level_to_db(level)


# Envelope rate (R11/R12 period → Hz) spans a huge dynamic range (~0.1 Hz to several kHz) and only a
# few frames per tune (re)write it, so a log-MSE regression is extremely noisy (its masked loss
# swings wildly batch-to-batch). Classify it over a log-spaced grid instead — the same fix that
# stabilised pitch/volume — giving a bounded cross-entropy with no blow-ups. Decode picks the
# arg-max bin centre (a representative Hz the compiler maps back to a period register).
ENV_RATE_MIN_HZ = ENV_RATE_FLOOR_HZ            # 0.1 Hz (matches the decode floor)
ENV_RATE_MAX_HZ = 8000.0                       # ~AY envelope step-rate ceiling at typical clocks
N_ENV_RATE_BINS = 48
_ENV_RATE_LOG_MIN = math.log(ENV_RATE_MIN_HZ)
_ENV_RATE_LOG_MAX = math.log(ENV_RATE_MAX_HZ)


def env_rate_to_bin(hz: float) -> int:
    """Quantise an envelope rate (Hz) to its nearest log-spaced bin in ``[0, N_ENV_RATE_BINS)``."""
    hz = min(max(hz, ENV_RATE_MIN_HZ), ENV_RATE_MAX_HZ)
    frac = (math.log(hz) - _ENV_RATE_LOG_MIN) / (_ENV_RATE_LOG_MAX - _ENV_RATE_LOG_MIN)
    b = round(frac * (N_ENV_RATE_BINS - 1))
    return int(min(max(b, 0), N_ENV_RATE_BINS - 1))


def bin_to_env_rate(b: int) -> float:
    """Inverse of :func:`env_rate_to_bin`: bin index → envelope rate in Hz (bin centre)."""
    frac = b / (N_ENV_RATE_BINS - 1)
    return math.exp(_ENV_RATE_LOG_MIN + frac * (_ENV_RATE_LOG_MAX - _ENV_RATE_LOG_MIN))


# Noise pitch (brightness) writes the 5-bit noise-period register R6 (NP in [1, 31]). A continuous
# regression jitters by a level every frame and churns R6 (it dominated the compiled stream), so we
# classify it over the 31 discrete noise-period values instead — same fix as pitch/volume/env_rate.
# The brightness↔period mapping is the compiler's canonical table (single source of truth).
N_NOISE_LEVELS = NP_MAX - NP_MIN + 1  # 31 distinct 5-bit noise-period values (1..31)


def noise_pitch_to_level(brightness: float) -> int:
    """Brightness 0..1 → noise-period class index in ``[0, N_NOISE_LEVELS)`` (compiler table)."""
    return noise_pitch_to_np(brightness) - NP_MIN


def noise_level_to_pitch(level: int) -> float:
    """Inverse: noise-period class index → brightness 0..1 via the compiler's noise-period table."""
    return np_to_noise_pitch(int(level) + NP_MIN)

