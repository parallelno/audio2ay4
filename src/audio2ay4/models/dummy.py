"""Dummy baseline core: a deterministic, untrained placeholder.

It is **not** a real solution — it exists so the full convert→compile→preview pipeline is runnable
and testable end-to-end before any model is trained. It maps each frame's dominant mel band to a
single tone on voice A, with volume tracking frame energy. Plan A / Plan B replace it.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import RunConfig
from ..features.mel import mel_center_freqs
from ..repr.compile import hz_to_semitones
from ..repr.state import AYGlobalFrame, AYState, AYStateFrame, AYVoiceFrame
from .base import register_core

_SILENT = AYVoiceFrame(pitch_semitones=float("nan"), volume_db=float("-inf"), tone_on=False)
_ENERGY_GATE = 0.05


class DummyCore:
    """Argmax-mel → single-voice tone. Deterministic, no learned parameters."""

    def infer(self, feats, cfg: RunConfig) -> AYState:
        f = np.asarray(feats.feats, dtype=np.float64)
        n, dim = f.shape
        centers = mel_center_freqs(n_mels=dim, sr=cfg.sample_rate)
        energy = f.sum(axis=1)
        e_max = float(energy.max()) or 1.0

        state: AYState = []
        for i in range(n):
            e = energy[i] / e_max
            if e < _ENERGY_GATE:
                voices = (_SILENT, _SILENT, _SILENT)
            else:
                k = int(np.argmax(f[i]))
                hz = float(centers[k]) if k < len(centers) else 440.0
                semi = hz_to_semitones(max(20.0, hz))
                volume_db = -36.0 * (1.0 - e)  # quieter when frame energy is low
                voice = AYVoiceFrame(
                    pitch_semitones=semi,
                    volume_db=volume_db if math.isfinite(volume_db) else 0.0,
                    tone_on=True,
                )
                voices = (voice, _SILENT, _SILENT)
            state.append(AYStateFrame(voices=voices, glob=AYGlobalFrame()))
        return state


@register_core("dummy")
def _make_dummy(cfg: RunConfig) -> DummyCore:
    return DummyCore()
