"""Eval harness: run a core on audio, render its YM, and score fidelity/stability/legality."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .. import chip, features, io
from ..config import RunConfig
from ..models import get_core
from ..repr import compile_state
from .metrics import spectral_distance, stability

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")


@dataclass
class EvalResult:
    """Scores for one converted track."""

    name: str
    spectral_distance: float
    stability: float
    legal: bool
    n_frames: int


def evaluate_audio(in_path: str, cfg: RunConfig) -> EvalResult:
    """Convert ``in_path`` with the configured core, render it back, and compare to the input."""
    audio = io.decode(in_path, target_sr=cfg.sample_rate, mono=True)
    feats = features.extract(audio, cfg)
    state = get_core(cfg.core, cfg).infer(feats, cfg)
    song = compile_state(state, cfg)
    rendered = chip.render_song(song, sample_rate=cfg.sample_rate)
    return EvalResult(
        name=os.path.basename(in_path),
        spectral_distance=spectral_distance(audio.pcm, rendered),
        stability=stability(song.regs),
        legal=bool(chip.is_legal(song.regs)),
        n_frames=song.n_frames,
    )


def evaluate_path(path: str, cfg: RunConfig) -> list[EvalResult]:
    """Evaluate a single audio file or every audio file in a directory."""
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(_AUDIO_EXTS)
        )
    else:
        files = [path]
    return [evaluate_audio(f, cfg) for f in files]


def aggregate(results: list[EvalResult]) -> dict[str, float]:
    """Summarise a batch of results into mean metrics + the legality pass-rate."""
    if not results:
        return {"count": 0.0, "spectral_distance": 0.0, "stability": 0.0, "legality_rate": 1.0}
    n = len(results)
    return {
        "count": float(n),
        "spectral_distance": sum(r.spectral_distance for r in results) / n,
        "stability": sum(r.stability for r in results) / n,
        "legality_rate": sum(1.0 for r in results if r.legal) / n,
    }
