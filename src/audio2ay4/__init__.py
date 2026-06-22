"""audio2ay4 — audio → AY-3-8910 / YM2149 register streams (.ym), with audio preview.

See ``design/`` for the architecture. This package is the shared **Milestone 0** foundation:
typed stage contracts, a deterministic register compiler, and a pluggable ``LearnedCore`` slot
that Plan A (RL) and Plan B (diffusion) implement. The proven chip emulator and YM I/O are reused
as-is from audio2ay3 (https://github.com/parallelno/audio2ay3) via a thin adapter.
"""

from .config import RunConfig, TrainConfig
from .repr.state import (
    AudioBuffer,
    FeatureFrames,
    AYVoiceFrame,
    AYGlobalFrame,
    AYStateFrame,
    AYState,
    YmSong,
)

__all__ = [
    "RunConfig",
    "TrainConfig",
    "AudioBuffer",
    "FeatureFrames",
    "AYVoiceFrame",
    "AYGlobalFrame",
    "AYStateFrame",
    "AYState",
    "YmSong",
]

__version__ = "0.0.1"
