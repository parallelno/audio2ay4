"""Score candidate audio renders against an original, using the perceptual metrics.

Usage:
    python scripts/score_candidates.py <original_audio> <candidate1> [<candidate2> ...]

Reports, per candidate (all vs. the SAME original):
  chroma  = pitch-class/melody agreement in [0,1]  (HIGHER = better, the headline number)
  onset   = rhythm/timing correlation in [-1,1]     (HIGHER = better)
  spec    = multi-res STFT-magnitude distance       (LOWER = better; the OLD, perceptually weak one)
"""

from __future__ import annotations

import sys

from audio2ay4 import io
from audio2ay4.eval import chroma_similarity, onset_similarity, spectral_distance

SR = 44_100


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    original = io.decode(argv[0], target_sr=SR, mono=True).pcm
    print(f"{'candidate':<40} {'chroma':>8} {'onset':>8} {'spec':>8}")
    rows = []
    for path in argv[1:]:
        cand = io.decode(path, target_sr=SR, mono=True).pcm
        ch = chroma_similarity(original, cand, SR)
        on = onset_similarity(original, cand, SR)
        sp = spectral_distance(original, cand)
        rows.append((path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1], ch, on, sp))
    for name, ch, on, sp in sorted(rows, key=lambda r: -r[1]):  # rank by chroma, best first
        print(f"{name:<40} {ch:>8.4f} {on:>8.4f} {sp:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
