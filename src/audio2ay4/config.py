"""Typed run/train configuration (see design/README.md §4, §6).

Kept dependency-free (plain dataclasses) so the deterministic core imports without torch/pydantic.
Swap to pydantic later if richer validation/serialization is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ZX Spectrum master clock; configurable (2.0 MHz Atari ST, 1.0 MHz CPC, 1.7897 MHz MSX).
DEFAULT_MASTER_CLOCK_HZ = 1_773_400
DEFAULT_FRAME_RATE_HZ = 50


@dataclass(frozen=True)
class RunConfig:
    """Inference-time configuration shared by every stage."""

    core: str = "dummy"                 # "dummy" | "rl" | "diffusion"
    feat_kind: str = "mel"              # "mel" | "cqt" | "encodec"
    master_clock_hz: int = DEFAULT_MASTER_CLOCK_HZ
    frame_rate_hz: int = DEFAULT_FRAME_RATE_HZ
    n_chips: int = 1                    # 1 = single AY; 2 = dual-AY (later)
    sample_rate: int = 44_100           # internal working / render rate
    use_gpu: bool = True
    seed: int = 0
    # Free-form backend options (model checkpoint paths, sampler steps, guidance weight, ...).
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    """Training-time configuration (Plan A / Plan B)."""

    plan: str = "rl"                    # "rl" | "diffusion"
    run: RunConfig = field(default_factory=RunConfig)
    batch_size: int = 16
    lr: float = 1e-4
    max_steps: int = 100_000
    corpus_dir: str = ""               # ingested YM corpus root (see data/)
    cache_dir: str = ".cache"
    extra: dict = field(default_factory=dict)
