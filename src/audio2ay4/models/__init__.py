"""Learned cores — the swappable brain that maps features → smooth ``AYState`` (design §4, §6).

Every core (the dummy baseline now; Plan A RL and Plan B diffusion later) implements the same
:class:`LearnedCore` protocol and is reached through :func:`get_core`, so the convert pipeline never
depends on a specific model. Plan A/B cores live behind the optional ``neural`` extra and are
registered lazily.
"""

from __future__ import annotations

from .base import LearnedCore, get_core, register_core

__all__ = ["LearnedCore", "get_core", "register_core"]
