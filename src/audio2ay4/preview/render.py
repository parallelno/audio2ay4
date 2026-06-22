"""Preview rendering — uses audio2ay3's emulator (ground truth) to sonify a YM stream."""

from __future__ import annotations

from .. import chip, io
from ..config import RunConfig
from ..convert import convert_audio_to_ym


def preview_ym(in_path: str, out_path: str, cfg: RunConfig) -> None:
    """Render an existing YM file to an audio file (``.wav`` always; others need ``[audio]``)."""
    song = chip.read_ym(in_path)
    pcm = chip.render_song(song, sample_rate=cfg.sample_rate)
    io.encode(out_path, pcm, cfg.sample_rate)


def preview_audio(in_path: str, out_path: str, cfg: RunConfig, ym_path: str | None = None) -> None:
    """Convert an audio file to YM, then render that YM back to audio (round-trip preview)."""
    import os
    import tempfile

    target_ym = ym_path or os.path.join(tempfile.gettempdir(), "audio2ay4_preview.ym")
    song = convert_audio_to_ym(in_path, target_ym, cfg)
    pcm = chip.render_song(song, sample_rate=cfg.sample_rate)
    io.encode(out_path, pcm, cfg.sample_rate)
