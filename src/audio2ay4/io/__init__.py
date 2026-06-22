"""Audio I/O: decode many container formats to PCM, encode previews back out (design §5).

soundfile (the ``audio`` extra) is used when available for broad format support; a stdlib ``wave``
fallback keeps ``.wav`` working in the light core. MP3 decode/encode requires the ``audio`` extra
(or ffmpeg) and raises a clear hint otherwise.
"""

from __future__ import annotations

from .audio import decode, encode

__all__ = ["decode", "encode"]
