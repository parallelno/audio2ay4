"""Plan A learned core (RL reverse player). Importing ``.core`` registers the ``rl`` core.

Lives behind the optional ``neural`` extra (torch). The torch-based ``RLCore``/``ReversePlayer`` are
exposed lazily so that torch-free modules can import ``policy.spec`` (the head specification) without
dragging in the torch stack — this keeps the parallel render workers numpy-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["RLCore", "ReversePlayer"]

if TYPE_CHECKING:  # type checkers only; no torch import at runtime
    from .core import RLCore
    from .network import ReversePlayer

_LAZY = {"RLCore": "core", "ReversePlayer": "network"}


def __getattr__(name: str):  # PEP 562 lazy attribute loading (defers torch import)
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)
