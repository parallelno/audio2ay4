"""YM → ``(audio features, ground-truth registers)`` pairing with a disk cache.

For each corpus YM we render audio via audio2ay3's emulator, extract frame-aligned features, and
keep the original registers as the supervised target. The expensive render+feature step is cached
to ``.npz`` keyed by ``(content sha1, feature config)`` — this is also Milestone-0 item 2's
"feature front-end with cached outputs".

We cache *registers* (not the smooth ``AYState``): they are plain ``uint8`` arrays, and the
``AYState`` target is reconstructed deterministically on demand via ``parse_song``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

from .. import chip, features
from ..config import RunConfig
from ..repr import parse_song
from ..repr.state import AudioBuffer, AYState, FeatureFrames, YmSong

_CACHE_VERSION = 1


@dataclass
class TrainingPair:
    """One supervised example: audio features ↔ the registers that produced that audio."""

    feats: FeatureFrames
    target_regs: np.ndarray   # (n_frames, 16) uint8 — ground truth
    master_clock_hz: int
    frame_rate_hz: int
    meta: dict

    @property
    def n_frames(self) -> int:
        return int(self.target_regs.shape[0])

    @property
    def target_state(self) -> AYState:
        """Decode the ground-truth registers into the smooth ``AYState`` a core should emit."""
        song = YmSong(
            regs=self.target_regs,
            master_clock_hz=self.master_clock_hz,
            frame_rate_hz=self.frame_rate_hz,
        )
        return parse_song(song)


def _cfg_key(cfg: RunConfig) -> str:
    raw = f"v{_CACHE_VERSION}:{cfg.sample_rate}:{cfg.frame_rate_hz}:{cfg.feat_kind}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _file_sha1(path: str, _buf: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(cache_dir: str, sha1: str, cfg: RunConfig) -> str:
    return os.path.join(cache_dir, f"{sha1}_{_cfg_key(cfg)}.npz")


def _load_cache(path: str) -> TrainingPair | None:
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        feats = FeatureFrames(
            feats=z["feats"].astype(np.float32),
            frame_rate=int(z["frame_rate"]),
            feat_kind=str(z["feat_kind"]),
        )
        return TrainingPair(
            feats=feats,
            target_regs=z["target_regs"].astype(np.uint8),
            master_clock_hz=int(z["master_clock_hz"]),
            frame_rate_hz=int(z["frame_rate"]),
            meta={"name": str(z["name"]), "author": str(z["author"])},
        )
    except Exception:
        return None  # treat a corrupt cache entry as a miss


def _save_cache(path: str, pair: TrainingPair) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        feats=pair.feats.feats.astype(np.float32),
        frame_rate=np.int64(pair.frame_rate_hz),
        feat_kind=np.str_(pair.feats.feat_kind),
        target_regs=pair.target_regs.astype(np.uint8),
        master_clock_hz=np.int64(pair.master_clock_hz),
        name=np.str_(pair.meta.get("name", "")),
        author=np.str_(pair.meta.get("author", "")),
    )


def build_pair(ym_path: str, cfg: RunConfig, cache_dir: str | None = None) -> TrainingPair:
    """Build (or load from cache) the training pair for ``ym_path``.

    Features and target registers are length-aligned to the shorter of the two (the rendered audio
    yields ~one feature frame per AY frame; we trim any off-by-one tail).
    """
    cache_file = None
    if cache_dir:
        cache_file = _cache_path(cache_dir, _file_sha1(ym_path), cfg)
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

    song = chip.read_ym(ym_path)
    pcm = chip.render_song(song, sample_rate=cfg.sample_rate)
    audio = AudioBuffer(pcm=pcm, sample_rate=cfg.sample_rate, duration_s=len(pcm) / float(cfg.sample_rate))
    feats = features.extract(audio, cfg)

    n = min(feats.n_frames, song.n_frames)
    feats = FeatureFrames(feats=feats.feats[:n], frame_rate=feats.frame_rate, feat_kind=feats.feat_kind)
    target_regs = np.ascontiguousarray(song.regs[:n], dtype=np.uint8)

    pair = TrainingPair(
        feats=feats,
        target_regs=target_regs,
        master_clock_hz=int(song.master_clock_hz),
        frame_rate_hz=int(song.frame_rate_hz),
        meta=song.meta or {},
    )
    if cache_file:
        _save_cache(cache_file, pair)
    return pair
