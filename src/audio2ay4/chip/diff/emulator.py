"""Differentiable AY-3-8910 / YM2149 emulator (design A.3, Regime 1).

A DDSP-style twin of the trusted ``audio2ay3`` emulator: band-limited square tones + LFSR-matched
noise + a 16-step envelope, mixed exactly as the hardware (active-low gating, ``/3`` average, 2x
oversample). Unlike the trusted emulator (an integer, branch-heavy, **non-differentiable**
reference), this renders with smooth torch ops so gradients flow from the rendered audio back into
the continuous control parameters — the prerequisite for analysis-by-synthesis reward training.

It is **not** a replacement for ``chip.render_song``: ``audio2ay3`` stays the ground-truth renderer
(final ``.ym`` output, eval) and the **oracle this module is validated against**. This twin trades
bit-exactness for differentiability — it matches the oracle *spectrally / perceptually*, which is
all the reward gradient needs.

Conventions mirror the oracle (datasheet formulae):
    f_tone  = master_clock / (16 * TP)
    f_noise = master_clock / (16 * NP)
    f_env   = master_clock / (256 * EP)   (per envelope step)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

# Measured (MAME-style) 16-level AY DAC curve. Mirrors ``audio2ay3.chip.volume_tables.AY_DAC`` so
# rendered amplitudes match the oracle exactly (verified by the diff-vs-oracle tests).
AY_DAC: tuple[float, ...] = (
    0.0000, 0.0076, 0.0110, 0.0158,
    0.0231, 0.0344, 0.0519, 0.0764,
    0.1170, 0.1632, 0.2392, 0.3536,
    0.5043, 0.6261, 0.8071, 1.0000,
)

_TWO_PI = 2.0 * np.pi


@dataclass
class DiffControls:
    """Per-frame, **continuous** control tensors the differentiable emulator synthesises from.

    All tensors share a leading ``(n_frames, ...)`` axis. Gates / ``use_env`` live in ``[0, 1]``
    (binary when unpacked from registers, soft during straight-through training); ``level`` is a
    continuous DAC index in ``[0, 15]``. This is the smooth space the policy emits — the integer
    register packing is bypassed so the path stays differentiable.
    """

    tone_period: Tensor   # (n, 3) 12-bit tone period TP (>= 1)
    noise_period: Tensor  # (n,)   5-bit noise period NP (>= 1)
    env_period: Tensor    # (n,)   16-bit envelope period EP (>= 1)
    tone_gate: Tensor     # (n, 3) in [0, 1] — tone audible per voice
    noise_gate: Tensor    # (n, 3) in [0, 1] — noise audible per voice
    level: Tensor         # (n, 3) fixed DAC level in [0, 15]
    use_env: Tensor       # (n, 3) in [0, 1] — voice follows the envelope
    env_shape: Tensor     # (n,)   long, R13 shape nibble 0..15
    env_retrig: Tensor    # (n,)   in [0, 1] — envelope (re)triggered this frame

    def to(self, device: torch.device | str) -> DiffControls:
        return DiffControls(
            tone_period=self.tone_period.to(device),
            noise_period=self.noise_period.to(device),
            env_period=self.env_period.to(device),
            tone_gate=self.tone_gate.to(device),
            noise_gate=self.noise_gate.to(device),
            level=self.level.to(device),
            use_env=self.use_env.to(device),
            env_shape=self.env_shape.to(device),
            env_retrig=self.env_retrig.to(device),
        )


def unpack_regs(regs: np.ndarray, *, dtype: torch.dtype = torch.float32) -> DiffControls:
    """Decode a ``(n_frames, 16)`` uint8 register stream into continuous :class:`DiffControls`.

    Non-differentiable bit unpacking (used to feed the twin the *same* input as the oracle for
    validation, and to bootstrap controls from a parsed corpus stream). Training drives
    :meth:`DiffAyEmulator.render` with policy-emitted controls directly.
    """
    r = np.ascontiguousarray(regs, dtype=np.uint8)
    if r.ndim != 2 or r.shape[1] < 16:
        raise ValueError(f"expected (n_frames, >=16) register array, got {r.shape}")
    n = r.shape[0]

    tp = np.empty((n, 3), np.float64)
    for c in range(3):
        tp[:, c] = ((r[:, 2 * c + 1].astype(np.uint16) & 0x0F) << 8) | r[:, 2 * c]
    tp = np.maximum(tp, 1.0)

    npv = np.maximum((r[:, 6] & 0x1F).astype(np.float64), 1.0)
    epv = np.maximum(((r[:, 12].astype(np.uint32) << 8) | r[:, 11]).astype(np.float64), 1.0)

    mixer = r[:, 7]
    tone_gate = np.stack([1.0 - ((mixer >> c) & 1) for c in range(3)], axis=1).astype(np.float64)
    noise_gate = np.stack([1.0 - ((mixer >> (3 + c)) & 1) for c in range(3)], axis=1).astype(
        np.float64
    )

    amp = r[:, 8:11].astype(np.uint8)
    use_env = ((amp >> 4) & 1).astype(np.float64)
    level = (amp & 0x0F).astype(np.float64)

    r13 = r[:, 13]
    retrig = (r13 != 0xFF).astype(np.float64)
    shape = np.where(r13 != 0xFF, r13 & 0x0F, 0).astype(np.int64)

    t = lambda a: torch.as_tensor(a, dtype=dtype)  # noqa: E731
    return DiffControls(
        tone_period=t(tp),
        noise_period=t(npv),
        env_period=t(epv),
        tone_gate=t(tone_gate),
        noise_gate=t(noise_gate),
        level=t(level),
        use_env=t(use_env),
        env_shape=torch.as_tensor(shape, dtype=torch.long),
        env_retrig=t(retrig),
    )


def _lfsr_sequence(length: int) -> np.ndarray:
    """The oracle's 17-bit LFSR output bit (taps 0 and 3), one entry per noise step."""
    out = np.empty(length, np.float32)
    lfsr = 1
    for i in range(length):
        out[i] = lfsr & 1
        bit = (lfsr ^ (lfsr >> 3)) & 1
        lfsr = ((lfsr >> 1) | (bit << 16)) & 0x1FFFF
    return out


class DiffAyEmulator(nn.Module):
    """Render :class:`DiffControls` (or a register stream) to mono PCM, differentiably.

    Mirrors ``audio2ay3.chip.Ay3Emulator``: internal render at ``render_sr * oversample`` then a
    box-average down to ``render_sr``. Tones are additive band-limited squares (odd harmonics below
    Nyquist), noise is the oracle's LFSR sample-held at ``f_noise``, and the envelope is the 16-step
    ramp/triangle/hold family with the DAC table interpolated for sub-step gradients.
    """

    dac: Tensor
    _noise: Tensor

    def __init__(
        self,
        render_sr: int = 44_100,
        oversample: int = 2,
        max_partials: int = 32,
        noise_len: int = 1 << 20,
    ) -> None:
        super().__init__()
        self.render_sr = int(render_sr)
        self.oversample = max(1, int(oversample))
        self.max_partials = max(1, int(max_partials))
        self.register_buffer("dac", torch.tensor(AY_DAC, dtype=torch.float32))
        self.register_buffer("_noise", torch.from_numpy(_lfsr_sequence(int(noise_len))))

    @property
    def internal_sr(self) -> float:
        return float(self.render_sr * self.oversample)

    def render_regs(self, regs: np.ndarray, master_clock: int, frame_rate: int) -> Tensor:
        """Convenience: unpack a uint8 register stream and render it (for oracle comparison)."""
        controls = unpack_regs(regs).to(self.dac.device)
        return self.render(controls, master_clock, frame_rate)

    def render(self, c: DiffControls, master_clock: int, frame_rate: int) -> Tensor:
        """Render continuous controls → mono float32 PCM ``(n_samples,)`` at ``render_sr``."""
        device = self.dac.device
        n_frames = c.tone_period.shape[0]
        if n_frames == 0:
            return torch.zeros(0, device=device)

        isr = self.internal_sr
        spf = isr / float(frame_rate)
        bounds = np.round(np.arange(n_frames + 1) * spf).astype(np.int64)
        total = int(bounds[-1])
        counts = np.diff(bounds)
        frame_idx = torch.from_numpy(np.repeat(np.arange(n_frames), counts)).to(device)
        starts = np.zeros(total, dtype=bool)
        starts[bounds[:-1]] = True
        is_start = torch.from_numpy(starts).to(device)

        def expand(x: Tensor) -> Tensor:
            return x.index_select(0, frame_idx)

        clock = float(master_clock)

        # --- tones: additive band-limited unipolar squares (mean 0.5), phase continuous ---------
        tone_period = expand(c.tone_period).clamp(min=1.0)          # (total, 3)
        f_tone = clock / (16.0 * tone_period)                       # (total, 3) Hz
        inc = (_TWO_PI * f_tone / isr).to(torch.float64)
        phi = torch.cumsum(inc, dim=0)
        phi = torch.remainder(phi, _TWO_PI).to(f_tone.dtype)        # wrap for float32 sin accuracy
        square = torch.full_like(f_tone, 0.5)
        nyq = isr / 2.0
        for j in range(self.max_partials):
            k = 2 * j + 1                                          # odd harmonics only
            mask = (k * f_tone < nyq).to(f_tone.dtype)            # drop partials past Nyquist
            square = square + (2.0 / np.pi) * mask * torch.sin(k * phi) / k
        tone_gate = expand(c.tone_gate)
        tone_term = tone_gate * square + (1.0 - tone_gate)         # gate off ⇒ pass-through 1

        # --- noise: oracle LFSR, sample-held at f_noise -----------------------------------------
        noise_period = expand(c.noise_period).clamp(min=1.0)       # (total,)
        f_noise = clock / (16.0 * noise_period)
        n_phase = torch.cumsum((f_noise / isr).to(torch.float64), dim=0)
        n_idx = torch.remainder(n_phase.floor().to(torch.long), self._noise.shape[0])
        noise_val = self._noise.index_select(0, n_idx).unsqueeze(1)  # (total, 1)
        noise_gate = expand(c.noise_gate)
        noise_term = noise_gate * noise_val + (1.0 - noise_gate)

        # --- envelope: resettable 16-step ramp/triangle/hold, DAC interpolated -------------------
        env_amp = self._envelope(c, frame_idx, is_start, expand, clock, isr)  # (total,)

        # --- per-voice amplitude, mix exactly like the oracle (s/3) -----------------------------
        level = expand(c.level).clamp(0.0, 15.0)
        fixed_amp = self._dac_interp(level)                        # (total, 3)
        use_env = expand(c.use_env)
        amp = use_env * env_amp.unsqueeze(1) + (1.0 - use_env) * fixed_amp
        signal = (amp * tone_term * noise_term).sum(dim=1) / 3.0   # (total,)

        return self._downsample(signal)

    def _envelope(
        self,
        c: DiffControls,
        frame_idx: Tensor,
        is_start: Tensor,
        expand,
        clock: float,
        isr: float,
    ) -> Tensor:
        """Continuous envelope DAC amplitude per internal sample (reset at retrigger frames)."""
        env_period = expand(c.env_period).clamp(min=1.0)
        step_inc = (clock / (256.0 * env_period) / isr).to(torch.float64)   # env steps / sample
        retrig = expand(c.env_retrig)
        reset = (is_start & (retrig > 0.5))                                 # restart points

        csum = torch.cumsum(step_inc, dim=0)
        # Subtract the running total captured at the most recent reset ⇒ position restarts at ~0.
        offset = torch.where(reset, csum - step_inc, torch.zeros_like(csum))
        offset = torch.cummax(offset, dim=0).values
        pos = (csum - offset).to(torch.float32)                            # continuous step >= 0

        shape = expand(c.env_shape)
        cont = ((shape >> 3) & 1).to(pos.dtype)
        att = ((shape >> 2) & 1).to(pos.dtype)
        alt = ((shape >> 1) & 1).to(pos.dtype)
        hold = (shape & 1).to(pos.dtype)

        q = pos - 16.0 * torch.floor(pos / 16.0)                           # within-segment [0,16)
        seg_odd = torch.remainder(torch.floor(pos / 16.0), 2.0)            # 0/1 per segment
        first = (pos < 16.0).to(pos.dtype)

        up = q
        down = 15.0 - q
        ramp_att = att * up + (1.0 - att) * down                          # first-segment ramp

        # cont == 0: ramp once, then silence.
        lvl_oneshot = first * ramp_att

        # cont == 1, hold == 1: ramp once, then hold high/low per (att xor alt).
        held = torch.where((att + alt) == 1.0, torch.full_like(pos, 15.0), torch.zeros_like(pos))
        lvl_hold = first * ramp_att + (1.0 - first) * held

        # cont == 1, hold == 0: repeat; alt toggles direction each segment (triangle).
        att_seg = torch.remainder(att + alt * seg_odd, 2.0)
        lvl_repeat = att_seg * up + (1.0 - att_seg) * down

        lvl_cont = hold * lvl_hold + (1.0 - hold) * lvl_repeat
        level = cont * lvl_cont + (1.0 - cont) * lvl_oneshot
        return self._dac_interp(level.clamp(0.0, 15.0))

    def _dac_interp(self, level: Tensor) -> Tensor:
        """Linear interpolation into the DAC table at a continuous level in ``[0, 15]``."""
        lo = torch.floor(level).clamp(0.0, 15.0)
        hi = (lo + 1.0).clamp(0.0, 15.0)
        frac = (level - lo).clamp(0.0, 1.0)
        lo_i = lo.to(torch.long)
        hi_i = hi.to(torch.long)
        return self.dac[lo_i] * (1.0 - frac) + self.dac[hi_i] * frac

    def _downsample(self, x: Tensor) -> Tensor:
        if self.oversample == 1 or x.numel() == 0:
            return x
        n = (x.shape[0] // self.oversample) * self.oversample
        if n == 0:
            return x.new_zeros(0)
        return x[:n].reshape(-1, self.oversample).mean(dim=1)
