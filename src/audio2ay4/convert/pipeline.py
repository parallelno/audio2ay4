"""End-to-end conversion: decode → features → core → compile → write YM.

Each stage depends only on the shared typed contracts, so any stage (core especially) is swappable
without touching this orchestration.
"""

from __future__ import annotations

from .. import chip, features, io
from ..config import RunConfig
from ..models import get_core
from ..repr import compile_state
from ..repr.state import YmSong


def convert_audio_to_ym(in_path: str, out_path: str, cfg: RunConfig) -> YmSong:
    """Convert an audio file to a YM register stream and write it to ``out_path``."""
    audio = io.decode(in_path, target_sr=cfg.sample_rate, mono=True)
    feats = features.extract(audio, cfg)
    core = get_core(cfg.core, cfg)
    state = core.infer(feats, cfg)
    song = compile_state(state, cfg)
    chip.write_ym(song, out_path)
    return song
