"""Parallel corpus rendering for the warm-start (torch-free).

``build_pair`` renders one YM through the emulator and extracts features — CPU-bound and entirely
independent per file, so it fans out across processes. This module is deliberately torch-free: the
process workers import only numpy + audio2ay3 + the (numpy) feature/target code, so spawning many
of them stays cheap in both startup time and memory.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..config import RunConfig
from ..data.pairing import TrainingPair, build_pair
from .targets import build_targets

Sample = tuple[np.ndarray, dict[str, np.ndarray]]


def pair_to_sample(pair: TrainingPair) -> Sample:
    """One training pair → ``(feats (T, dim), targets)`` for collation."""
    feats = np.asarray(pair.feats.feats, dtype=np.float32)
    targets = build_targets(pair.target_regs, pair.master_clock_hz, pair.frame_rate_hz)
    n = min(feats.shape[0], targets["env_shape"].shape[0])
    feats = feats[:n]
    targets = {k: v[..., :n] if v.ndim == 1 else v[:, :n] for k, v in targets.items()}
    return feats, targets


def _render_one(task: tuple[str, RunConfig, str | None]) -> Sample:
    """Process-pool worker: render one YM and reduce it to a training sample."""
    path, run, cache_dir = task
    return pair_to_sample(build_pair(path, run, cache_dir=cache_dir))


def resolve_workers(workers: int | None, total: int) -> int:
    """0/None → all CPUs; always clamped to ``[1, total]``."""
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    return max(1, min(workers, total))


def render_samples(
    ym_paths: list[str],
    run: RunConfig,
    cache_dir: str | None,
    *,
    workers: int | None = None,
    log_every: int = 100,
) -> list[Sample]:
    """Render every YM to a training sample, skipping unparseable files.

    Rendering is parallelised across ``workers`` processes (default: all CPUs). Output order matches
    ``ym_paths`` regardless of completion order, so a fixed seed still gives reproducible batches.
    """
    total = len(ym_paths)
    n_workers = resolve_workers(workers, total)
    print(f"Rendering {total} YM files (workers={n_workers}) …", flush=True)

    results: dict[int, Sample] = {}
    t0 = time.monotonic()
    done = 0

    def _tick() -> None:
        elapsed = time.monotonic() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"  rendered {done}/{total} ({100 * done // total}%) "
            f"| ok {len(results)} | {rate:.1f} f/s | ETA {eta:.0f}s",
            flush=True,
        )

    if n_workers <= 1:
        for i, path in enumerate(ym_paths):
            try:
                results[i] = _render_one((path, run, cache_dir))
            except Exception as exc:  # noqa: BLE001 — one bad tune must not kill the run
                print(f"[skip] {path}: {exc}", file=sys.stderr, flush=True)
            done += 1
            if done % log_every == 0 or done == total:
                _tick()
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_render_one, (p, run, cache_dir)): i
                       for i, p in enumerate(ym_paths)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # noqa: BLE001 — skip & log, keep going
                    print(f"[skip] {ym_paths[i]}: {exc}", file=sys.stderr, flush=True)
                done += 1
                if done % log_every == 0 or done == total:
                    _tick()

    return [results[i] for i in sorted(results)]
