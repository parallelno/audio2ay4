"""Stage contracts — the typed data passed between stages (design/README.md §4).

These dataclasses are the project's stable interfaces: every stage is testable and swappable
because it only depends on these types, not on its neighbours' internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# --- Audio + features -----------------------------------------------------------------------


@dataclass
class AudioBuffer:
    """Decoded PCM (io.decode output)."""

    pcm: NDArray[np.float32]   # shape (samples, channels)
    sample_rate: int
    duration_s: float


@dataclass
class FeatureFrames:
    """Frame-aligned audio features at ``frame_rate`` (features.* output)."""

    feats: NDArray[np.float32]   # shape (frames, feat_dim)
    frame_rate: int              # e.g. 50
    feat_kind: str               # "mel" | "cqt" | "encodec"

    @property
    def n_frames(self) -> int:
        return int(self.feats.shape[0])


# --- AYState: the smooth intermediate representation the learned core emits ------------------


@dataclass
class AYVoiceFrame:
    """One tone channel for one frame, in perceptual units (pre-quantisation)."""

    pitch_semitones: float    # MIDI-like; NaN/-inf ⇒ silent
    volume_db: float          # 0 dB ≈ full; -inf ⇒ silent
    tone_on: bool = True
    noise_on: bool = False
    use_envelope: bool = False


@dataclass
class AYGlobalFrame:
    """The SHARED resources (single envelope, single noise) — modelled once per frame."""

    noise_pitch: float = 0.0  # 0..1 brightness control → 5-bit noise period
    env_shape: int = 0        # 0..15 (R13)
    env_rate: float = 1.0     # Hz → 16-bit envelope period
    env_retrigger: bool = False


@dataclass
class AYStateFrame:
    voices: tuple[AYVoiceFrame, AYVoiceFrame, AYVoiceFrame]
    glob: AYGlobalFrame


# A whole song is just a list of frames (length = n_frames).
AYState = list[AYStateFrame]


# --- Registers + song -----------------------------------------------------------------------

# Raw AY register snapshots: shape (frames, 16), dtype uint8.
RegisterStream = NDArray[np.uint8]


@dataclass
class YmSong:
    """Hardware-legal register stream + metadata; consumed by the YM writer or the emulator."""

    regs: RegisterStream          # (frames, 16) uint8
    master_clock_hz: int
    frame_rate_hz: int
    loop_frame: Optional[int] = None
    meta: Optional[dict] = None

    @property
    def n_frames(self) -> int:
        return int(self.regs.shape[0])


def silent_state(n_frames: int) -> AYState:
    """A legal, silent AYState of ``n_frames`` — handy placeholder / test fixture."""
    out: AYState = []
    for _ in range(n_frames):
        voices = tuple(
            AYVoiceFrame(pitch_semitones=float("nan"), volume_db=float("-inf"), tone_on=False)
            for _ in range(3)
        )
        out.append(AYStateFrame(voices=voices, glob=AYGlobalFrame()))  # type: ignore[arg-type]
    return out
