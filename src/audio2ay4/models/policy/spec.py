"""Shared head specification for the Plan A reverse player (torch-free).

Single source of truth for how the network's raw heads map to AYState units, imported by both the
torch side (``network``/``core``/training ``loss``) and the numpy side (training ``targets``). Kept
free of torch so the deterministic core and the target builder import without the neural extra.
"""

from __future__ import annotations

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
