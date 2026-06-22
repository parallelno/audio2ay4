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
