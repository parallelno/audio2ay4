"""Training subsystem (Plan A / Plan B).

``targets`` (numpy) is torch-free and imported eagerly; the torch-dependent warm-start loop and
loss are exposed lazily so ``import audio2ay4.train`` works without the optional ``neural`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .targets import build_targets

__all__ = [
    "build_targets",
    "train_warmstart",
    "train_step",
    "collate",
    "pair_to_sample",
    "warmstart_loss",
    "WarmstartWeights",
]

if TYPE_CHECKING:  # for type checkers only; no torch import at runtime
    from .warmstart import collate, pair_to_sample, train_step, train_warmstart
    from .warmstart_loss import WarmstartWeights, warmstart_loss

_LAZY = {
    "train_warmstart": "warmstart",
    "train_step": "warmstart",
    "collate": "warmstart",
    "pair_to_sample": "warmstart",
    "warmstart_loss": "warmstart_loss",
    "WarmstartWeights": "warmstart_loss",
}


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)
