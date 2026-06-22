"""Evaluation layer — perceptual/structural metrics + regression harness (design §5.5).

All metrics are numpy-only. The harness renders a core's output through audio2ay3's emulator (the
same ground truth used for training) and scores it against the input, so "what we measure" is
"what a chip produces".
"""

from __future__ import annotations

from .harness import EvalResult, aggregate, evaluate_audio, evaluate_path
from .metrics import legality_rate, spectral_distance, stability

__all__ = [
    "spectral_distance",
    "stability",
    "legality_rate",
    "EvalResult",
    "evaluate_audio",
    "evaluate_path",
    "aggregate",
]
