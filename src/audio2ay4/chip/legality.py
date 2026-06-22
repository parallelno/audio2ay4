"""Pure, dependency-free hardware-legality checks for a register stream.

This is the test oracle for the compiler (design/README.md §5.1): it asserts that a ``(n, 16)``
register array only contains states a real AY-3-8910 / YM2149 can hold. It has **no** dependency
on audio2ay3, so legality tests run in the light core environment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ENV_NO_WRITE = 0xFF  # ST-Sound R13 sentinel: "do not write / do not retrigger this frame"


def is_legal(regs: NDArray[np.uint8]) -> bool:
    """Return ``True`` iff every frame in ``regs`` is a legal AY register snapshot."""
    a = np.asarray(regs)
    if a.ndim != 2 or a.shape[1] != 16:
        return False
    if a.size == 0:
        return True
    if a.min() < 0 or a.max() > 255:
        return False
    a = a.astype(np.uint16)
    # Tone period high bytes (R1/R3/R5) are 12-bit ⇒ top nibble must be zero.
    if np.any(a[:, [1, 3, 5]] & 0xF0):
        return False
    # Noise period (R6) is 5-bit.
    if np.any(a[:, 6] & 0xE0):
        return False
    # Volume regs (R8/R9/R10): bit4 = envelope mode, bits0-3 = level ⇒ bits5-7 must be zero.
    if np.any(a[:, [8, 9, 10]] & 0xE0):
        return False
    # Envelope shape (R13): low nibble, or the 0xFF "no write" sentinel.
    r13 = a[:, 13]
    if np.any((r13 != ENV_NO_WRITE) & (r13 & 0xF0).astype(bool)):
        return False
    return True
