"""Data-pairing tests (emulator-gated): YM → (features, target registers), with caching."""

from __future__ import annotations

import os

import numpy as np
import pytest

from audio2ay4 import chip
from audio2ay4.chip.legality import is_legal
from audio2ay4.config import RunConfig
from audio2ay4.data import build_pair
from audio2ay4.data.pairing import _cache_path, _file_sha1
from audio2ay4.repr import compile_state
from audio2ay4.repr.state import AYGlobalFrame, AYStateFrame, AYVoiceFrame


def _make_ym(path: str, cfg: RunConfig, n: int = 40) -> None:
    state = []
    for i in range(n):
        v = AYVoiceFrame(pitch_semitones=48 + (i % 24), volume_db=-3.0 * (i % 4), tone_on=True)
        silent = AYVoiceFrame(pitch_semitones=float("nan"), volume_db=float("-inf"), tone_on=False)
        state.append(AYStateFrame(voices=(v, silent, silent), glob=AYGlobalFrame()))
    song = compile_state(state, cfg)
    chip.write_ym(song, path)


def test_build_pair_aligns_and_targets_legal(tmp_path):
    pytest.importorskip("audio2ay3")
    cfg = RunConfig(core="dummy")
    ym = tmp_path / "tune.ym"
    _make_ym(str(ym), cfg, n=40)

    pair = build_pair(str(ym), cfg)
    assert pair.feats.n_frames == pair.target_regs.shape[0]
    assert pair.target_regs.dtype == np.uint8
    assert is_legal(pair.target_regs)
    assert len(pair.target_state) == pair.n_frames


def test_build_pair_cache_roundtrip(tmp_path):
    pytest.importorskip("audio2ay3")
    cfg = RunConfig(core="dummy")
    ym = tmp_path / "tune.ym"
    cache = tmp_path / "cache"
    _make_ym(str(ym), cfg, n=24)

    first = build_pair(str(ym), cfg, cache_dir=str(cache))
    expected = _cache_path(str(cache), _file_sha1(str(ym)), cfg)
    assert os.path.exists(expected)

    second = build_pair(str(ym), cfg, cache_dir=str(cache))  # served from cache
    assert np.array_equal(first.target_regs, second.target_regs)
    assert np.allclose(first.feats.feats, second.feats.feats)
