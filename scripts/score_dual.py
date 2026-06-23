"""Render a dual-chip audio2ay3 result (chip1 + chip2 mixed) and score it vs the original.

Usage:
    python scripts/score_dual.py <original_audio> <chip1.ym> <chip2.ym> <out_mix.wav>

Prints chroma/onset for chip1-only and for the chip1+chip2 mix, so the dual-chip output is
scored fairly (percussion that lives on chip 2 is included).
"""

from __future__ import annotations

import sys

import numpy as np

from audio2ay4 import chip, io
from audio2ay4.eval import chroma_similarity, onset_similarity

SR = 44_100


def _render(ym_path: str) -> np.ndarray:
    return np.asarray(chip.render_song(chip.read_ym(ym_path), sample_rate=SR), dtype=np.float32)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    original_path, ym1_path, ym2_path, out_wav = argv
    original = io.decode(original_path, target_sr=SR, mono=True).pcm
    p1 = _render(ym1_path)
    p2 = _render(ym2_path)
    n = min(len(p1), len(p2))
    mix = p1[:n] + p2[:n]
    peak = float(np.max(np.abs(mix))) if n else 0.0
    if peak > 1.0:
        mix = mix / peak
    io.encode(out_wav, mix.astype(np.float32), SR)

    name = original_path.replace("\\", "/").rsplit("/", 1)[-1]
    print(f"{name:<28} {'chroma':>8} {'onset':>8}")
    print(f"  chip1 only{'':<16} {chroma_similarity(original, p1, SR):>8.4f} "
          f"{onset_similarity(original, p1, SR):>8.4f}")
    print(f"  chip1+chip2 mix{'':<11} {chroma_similarity(original, mix, SR):>8.4f} "
          f"{onset_similarity(original, mix, SR):>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
