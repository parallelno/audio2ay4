"""Chip core — reuses the **proven AY-3-8910 / YM2149 emulator and YM I/O from audio2ay3**.

audio2ay4 deliberately does **not** reinvent the emulator. This package is a thin adapter that
bridges audio2ay4's internal :class:`~audio2ay4.repr.state.YmSong` contract to audio2ay3's chip
core (the trusted ground truth for training, evaluation, and preview). See
https://github.com/parallelno/audio2ay3.

Pure, dependency-free legality checks live in :mod:`audio2ay4.chip.legality` so tests can run
without audio2ay3 installed.
"""

from __future__ import annotations

from .adapter import read_ym, render_song, write_ym
from .legality import is_legal

__all__ = ["render_song", "read_ym", "write_ym", "is_legal"]
