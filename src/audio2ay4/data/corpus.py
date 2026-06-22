"""Corpus ingest: discover, content-dedup, and read metadata for YM files in a directory tree."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from .. import chip


@dataclass(frozen=True)
class CorpusEntry:
    """One unique YM tune in the corpus (deduplicated by content hash)."""

    path: str
    sha1: str
    n_frames: int
    master_clock_hz: int
    frame_rate_hz: int
    name: str = ""
    author: str = ""

    @property
    def duration_s(self) -> float:
        return self.n_frames / float(self.frame_rate_hz or 50)


def _sha1(path: str, _buf: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def find_ym_files(root: str, recursive: bool = True) -> list[str]:
    """Return all ``*.ym`` paths under ``root`` (case-insensitive), sorted for determinism."""
    out: list[str] = []
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".ym"):
                    out.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            full = os.path.join(root, f)
            if os.path.isfile(full) and f.lower().endswith(".ym"):
                out.append(full)
    return sorted(out)


def scan_corpus(root: str, recursive: bool = True) -> list[CorpusEntry]:
    """Scan ``root`` for YM files, dedup by content, and read metadata via the YM reader.

    Unreadable files are skipped (corpora are noisy); duplicate content keeps the first path seen.
    Requires audio2ay3 (YM reader) — see :mod:`audio2ay4.chip`.
    """
    seen: set[str] = set()
    entries: list[CorpusEntry] = []
    for path in find_ym_files(root, recursive=recursive):
        try:
            digest = _sha1(path)
            if digest in seen:
                continue
            song = chip.read_ym(path)
        except Exception:
            continue  # skip corrupt / unsupported files
        seen.add(digest)
        meta = song.meta or {}
        entries.append(
            CorpusEntry(
                path=path,
                sha1=digest,
                n_frames=song.n_frames,
                master_clock_hz=int(song.master_clock_hz),
                frame_rate_hz=int(song.frame_rate_hz),
                name=str(meta.get("name", "")),
                author=str(meta.get("author", "")),
            )
        )
    return entries
