"""Data layer — shared training infrastructure for both plans (design/README.md §5.4).

Turns the public YM corpus into the **free, unlimited supervised signal** both Plan A and Plan B
build on: scan/dedup YM files (:func:`scan_corpus`), split by tune with no leakage
(:func:`split_by_tune`), and produce ``(audio features, ground-truth registers)`` pairs by
rendering each YM through audio2ay3's proven emulator (:func:`build_pair`).

Everything here is numpy-only except the lazy audio2ay3 dependency used to read/render YM files.
"""

from __future__ import annotations

from .corpus import CorpusEntry, scan_corpus
from .pairing import TrainingPair, build_pair
from .splits import split_by_tune

__all__ = ["CorpusEntry", "scan_corpus", "split_by_tune", "TrainingPair", "build_pair"]
