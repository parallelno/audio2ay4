"""Pure tests for deterministic by-tune corpus splitting."""

from __future__ import annotations

import hashlib

from audio2ay4.data import split_by_tune
from audio2ay4.data.corpus import CorpusEntry


def _entries(n: int) -> list[CorpusEntry]:
    out = []
    for i in range(n):
        sha = hashlib.sha1(f"tune-{i}".encode()).hexdigest()
        out.append(CorpusEntry(path=f"/c/{i}.ym", sha1=sha, n_frames=100 + i,
                               master_clock_hz=1_773_400, frame_rate_hz=50))
    return out


def test_splits_are_deterministic():
    entries = _entries(200)
    a = split_by_tune(entries, seed=7)
    b = split_by_tune(entries, seed=7)
    assert {k: [e.sha1 for e in v] for k, v in a.items()} == \
           {k: [e.sha1 for e in v] for k, v in b.items()}


def test_splits_cover_all_without_overlap():
    entries = _entries(300)
    s = split_by_tune(entries)
    seen = [e.sha1 for v in s.values() for e in v]
    assert sorted(seen) == sorted(e.sha1 for e in entries)
    assert len(seen) == len(set(seen))  # no leakage across splits


def test_splits_respect_ratios_roughly():
    entries = _entries(2000)
    s = split_by_tune(entries, ratios=(0.8, 0.1, 0.1), seed=0)
    frac = {k: len(v) / len(entries) for k, v in s.items()}
    assert abs(frac["train"] - 0.8) < 0.05
    assert abs(frac["val"] - 0.1) < 0.03
    assert abs(frac["test"] - 0.1) < 0.03


def test_seed_changes_partition():
    entries = _entries(500)
    a = split_by_tune(entries, seed=1)["train"]
    b = split_by_tune(entries, seed=2)["train"]
    assert [e.sha1 for e in a] != [e.sha1 for e in b]
