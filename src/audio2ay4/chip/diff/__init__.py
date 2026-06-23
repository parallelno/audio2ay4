"""Differentiable AY emulator (design A.3, Regime 1) — torch-only, imported lazily during training.

Kept out of :mod:`audio2ay4.chip`'s top-level exports so importing the chip package stays torch-free
(the trusted ``audio2ay3`` renderer and legality checks have no torch dependency).
"""

from __future__ import annotations

from .emulator import AY_DAC, DiffAyEmulator, DiffControls, unpack_regs

__all__ = ["DiffAyEmulator", "DiffControls", "unpack_regs", "AY_DAC"]
