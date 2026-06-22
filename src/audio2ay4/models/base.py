"""Core protocol + registry. Cores are looked up by name from :class:`RunConfig.core`."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from ..config import RunConfig
from ..repr.state import AYState, FeatureFrames


@runtime_checkable
class LearnedCore(Protocol):
    """Maps frame-aligned features to a smooth ``AYState`` (one state frame per feature frame)."""

    def infer(self, feats: FeatureFrames, cfg: RunConfig) -> AYState: ...


_REGISTRY: dict[str, Callable[[RunConfig], LearnedCore]] = {}


def register_core(name: str) -> Callable[[Callable[[RunConfig], LearnedCore]], Callable[[RunConfig], LearnedCore]]:
    """Decorator registering a ``factory(cfg) -> LearnedCore`` under ``name``."""

    def deco(factory: Callable[[RunConfig], LearnedCore]) -> Callable[[RunConfig], LearnedCore]:
        _REGISTRY[name] = factory
        return factory

    return deco


def get_core(name: str, cfg: RunConfig) -> LearnedCore:
    """Instantiate a registered core by name, importing built-ins on first use."""
    if not _REGISTRY:
        _bootstrap()
    if name not in _REGISTRY:
        if name in ("rl", "diffusion"):
            raise NotImplementedError(
                f"Core '{name}' is not implemented yet. See the design plan "
                f"(design/plan-{'a-reinforcement-learning' if name == 'rl' else 'b-diffusion'}.md)."
            )
        raise KeyError(f"Unknown core '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)


def _bootstrap() -> None:
    from . import dummy  # noqa: F401  (registers 'dummy' on import)
