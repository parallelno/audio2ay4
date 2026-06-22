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
_BOOTSTRAPPED = False


def register_core(name: str) -> Callable[[Callable[[RunConfig], LearnedCore]], Callable[[RunConfig], LearnedCore]]:
    """Decorator registering a ``factory(cfg) -> LearnedCore`` under ``name``."""

    def deco(factory: Callable[[RunConfig], LearnedCore]) -> Callable[[RunConfig], LearnedCore]:
        _REGISTRY[name] = factory
        return factory

    return deco


def get_core(name: str, cfg: RunConfig) -> LearnedCore:
    """Instantiate a registered core by name, importing built-ins on first use."""
    _bootstrap()
    if name == "rl" and name not in _REGISTRY:
        try:
            from . import policy  # noqa: F401  (registers 'rl' on import)
        except ImportError as exc:
            raise NotImplementedError(
                "The 'rl' core requires the neural extra (torch). "
                'Install it with: pip install -e ".[neural]".'
            ) from exc
    if name not in _REGISTRY:
        if name == "diffusion":
            raise NotImplementedError(
                "Core 'diffusion' is not implemented yet. See the design plan "
                "(design/plan-b-diffusion.md)."
            )
        raise KeyError(f"Unknown core '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)


def _bootstrap() -> None:
    """Register built-in cores once (idempotent; import order independent)."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    from . import dummy  # noqa: F401  (registers 'dummy' on import)

    _BOOTSTRAPPED = True
