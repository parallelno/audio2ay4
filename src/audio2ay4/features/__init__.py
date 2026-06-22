"""Audio feature front-end: decoded PCM → frame-aligned features at ``frame_rate`` (design §5.2).

The default is a numpy-only **log-mel spectrogram** so the core stays dependency-light. CQT /
EnCodec adapters can register here later behind the same ``extract`` entry point. Output is the
shared :class:`FeatureFrames` contract, aligned 1:1 with the AY frame grid that downstream stages
(and the learned core) consume.
"""

from __future__ import annotations

from .mel import extract, mel_center_freqs

__all__ = ["extract", "mel_center_freqs"]
