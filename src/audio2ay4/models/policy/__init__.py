"""Plan A learned core (RL reverse player). Importing this package registers the ``rl`` core.

Lives behind the optional ``neural`` extra (torch); imported lazily by ``models.get_core('rl')`` so
the deterministic core stays numpy-only importable.
"""

from __future__ import annotations

from .core import RLCore  # noqa: F401  (registers 'rl' via @register_core on import)
from .network import ReversePlayer

__all__ = ["RLCore", "ReversePlayer"]
