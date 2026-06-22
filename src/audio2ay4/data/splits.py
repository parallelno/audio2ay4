"""Deterministic train/val/test splitting **by tune** — no song leaks across splits (design §5.4).

The bucket is a stable hash of the content sha1 (already deduplicated), so the same corpus + seed
always yields the same split, independent of file order or filesystem.
"""

from __future__ import annotations

import hashlib

from .corpus import CorpusEntry

_SPLITS = ("train", "val", "test")


def _bucket(sha1: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{sha1}".encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 1_000_000) / 1_000_000.0


def split_by_tune(
    entries: list[CorpusEntry],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[CorpusEntry]]:
    """Partition ``entries`` into train/val/test by a stable per-tune hash.

    ``ratios`` need not sum to exactly 1; they are normalised. Result is deterministic for a given
    ``seed``.
    """
    total = float(sum(ratios)) or 1.0
    train_r, val_r, _ = (r / total for r in ratios)
    out: dict[str, list[CorpusEntry]] = {s: [] for s in _SPLITS}
    for e in entries:
        b = _bucket(e.sha1, seed)
        if b < train_r:
            out["train"].append(e)
        elif b < train_r + val_r:
            out["val"].append(e)
        else:
            out["test"].append(e)
    return out
