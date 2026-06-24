"""Audio feature front-end: decoded PCM → frame-aligned features at ``frame_rate`` (design §5.2).

The default is a numpy-only **log-mel spectrogram** so the core stays dependency-light. CQT /
EnCodec adapters can register here later behind the same ``extract`` entry point. Output is the
shared :class:`FeatureFrames` contract, aligned 1:1 with the AY frame grid that downstream stages
(and the learned core) consume.
"""

from __future__ import annotations

from ..config import RunConfig
from ..repr.state import AudioBuffer, FeatureFrames
from .mel import extract as _extract_mel
from .mel import mel_center_freqs


def extract(audio: AudioBuffer, cfg: RunConfig) -> FeatureFrames:
    """Audio → frame-aligned features, dispatched on ``cfg.feat_kind`` ("mel" | "cqt")."""
    if cfg.feat_kind == "cqt":
        from .cqt import extract as _extract_cqt
        return _extract_cqt(audio, cfg)
    return _extract_mel(audio, cfg)


__all__ = ["extract", "mel_center_freqs"]
