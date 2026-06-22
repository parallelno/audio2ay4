"""Deterministic register compiler: ``AYState`` → hardware-legal :class:`YmSong`.

This is the single choke point that guarantees every emitted frame is legal on a real
AY-3-8910 / YM2149 (design/README.md §2 principle 2, §5.3). No learned core produces register
values directly — cores emit the smooth :class:`AYState`, and this pure module quantises it,
arbitrates the **shared** envelope/noise generators, and writes registers.

The inverse, :func:`parse_song`, decodes a register stream back to ``AYState`` (used to build
training targets from corpus YM files).
"""

from __future__ import annotations

import math

import numpy as np

from ..config import RunConfig
from .state import AYGlobalFrame, AYState, AYStateFrame, AYVoiceFrame, YmSong

# Standard 16-level AY normalised amplitude table (approx; a *measured* DAC table replaces this
# in a later phase — design/README.md §5.3). Monotonic, logarithmic, [0, 1].
_AY_AMP = np.array(
    [0.0000, 0.0137, 0.0205, 0.0291, 0.0423, 0.0618, 0.0847, 0.1369,
     0.1691, 0.2647, 0.3527, 0.4499, 0.5704, 0.6873, 0.8482, 1.0000],
    dtype=np.float64,
)

TP_MIN, TP_MAX = 1, 4095        # 12-bit tone period
NP_MIN, NP_MAX = 1, 31          # 5-bit noise period
EP_MIN, EP_MAX = 1, 65535       # 16-bit envelope period
ENV_NO_WRITE = 0xFF


# --- scalar conversions ---------------------------------------------------------------------

def semitones_to_hz(semi: float) -> float:
    return 440.0 * (2.0 ** ((semi - 69.0) / 12.0))


def hz_to_semitones(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def hz_to_tp(hz: float, master_clock: int) -> int | None:
    """Frequency → 12-bit tone period, **octave-folding** content that is out of range."""
    if hz <= 0 or not math.isfinite(hz):
        return None
    tp = round(master_clock / (16.0 * hz))
    while tp > TP_MAX:        # too low to represent ⇒ fold up an octave
        hz *= 2.0
        tp = round(master_clock / (16.0 * hz))
    return int(max(TP_MIN, tp))


def db_to_level(db: float) -> int:
    """Perceptual dB (0 dB ≈ full) → nearest 4-bit DAC level via the amplitude table."""
    if not math.isfinite(db):
        return 0
    amp = min(1.0, max(0.0, 10.0 ** (db / 20.0)))
    return int(np.argmin(np.abs(_AY_AMP - amp)))


def level_to_db(level: int) -> float:
    amp = _AY_AMP[int(max(0, min(15, level)))]
    return 20.0 * math.log10(amp) if amp > 0 else float("-inf")


def noise_pitch_to_np(brightness: float) -> int:
    """Brightness 0..1 (1 = brightest) → 5-bit noise period (1 = highest freq)."""
    b = min(1.0, max(0.0, brightness))
    return int(min(NP_MAX, max(NP_MIN, round(NP_MIN + (1.0 - b) * (NP_MAX - NP_MIN)))))


def np_to_noise_pitch(npv: int) -> float:
    npv = int(max(NP_MIN, min(NP_MAX, npv)))
    return 1.0 - (npv - NP_MIN) / (NP_MAX - NP_MIN)


def env_rate_to_ep(hz: float, master_clock: int) -> int:
    if hz <= 0 or not math.isfinite(hz):
        return EP_MIN
    ep = round(master_clock / (256.0 * hz))
    return int(min(EP_MAX, max(EP_MIN, ep)))


def ep_to_env_rate(ep: int, master_clock: int) -> float:
    ep = int(max(EP_MIN, min(EP_MAX, ep)))
    return master_clock / (256.0 * ep)


# --- compile (AYState → registers) ----------------------------------------------------------

def compile_state(state: AYState, cfg: RunConfig) -> YmSong:
    """Compile a smooth ``AYState`` into a hardware-legal :class:`YmSong`."""
    n = len(state)
    regs = np.zeros((n, 16), dtype=np.uint8)
    mclk = int(cfg.master_clock_hz)

    for i, frame in enumerate(state):
        mixer = 0x3F  # active-low: bits0-5 set ⇒ all tone+noise disabled
        for c, v in enumerate(frame.voices):
            level = 0x10 if v.use_envelope else db_to_level(v.volume_db)
            audible = v.use_envelope or (level & 0x0F) > 0
            if v.tone_on and audible:
                tp = hz_to_tp(semitones_to_hz(v.pitch_semitones), mclk)
                if tp is not None:
                    regs[i, 2 * c] = tp & 0xFF
                    regs[i, 2 * c + 1] = (tp >> 8) & 0x0F
                    mixer &= ~(1 << c) & 0xFF        # enable tone
            if v.noise_on and audible:
                mixer &= ~(1 << (3 + c)) & 0xFF      # enable noise
            regs[i, 8 + c] = level & 0x1F

        regs[i, 7] = mixer

        g = frame.glob
        regs[i, 6] = noise_pitch_to_np(g.noise_pitch) & 0x1F
        ep = env_rate_to_ep(g.env_rate, mclk)
        regs[i, 11] = ep & 0xFF
        regs[i, 12] = (ep >> 8) & 0xFF
        regs[i, 13] = (g.env_shape & 0x0F) if g.env_retrigger else ENV_NO_WRITE

    return YmSong(
        regs=regs,
        master_clock_hz=mclk,
        frame_rate_hz=int(cfg.frame_rate_hz),
        loop_frame=None,
        meta=None,
    )


# --- parse (registers → AYState) ------------------------------------------------------------

def parse_song(song: YmSong) -> AYState:
    """Decode a register stream back into ``AYState`` (lossy inverse of :func:`compile_state`)."""
    regs = np.asarray(song.regs, dtype=np.uint8)
    mclk = int(song.master_clock_hz)
    out: AYState = []

    for i in range(regs.shape[0]):
        mixer = int(regs[i, 7])
        voices = []
        for c in range(3):
            tp = ((int(regs[i, 2 * c + 1]) & 0x0F) << 8) | int(regs[i, 2 * c])
            tone_on = ((mixer >> c) & 1) == 0
            noise_on = ((mixer >> (3 + c)) & 1) == 0
            vol = int(regs[i, 8 + c])
            use_env = bool(vol & 0x10)
            level = vol & 0x0F
            hz = mclk / (16.0 * max(1, tp))
            voices.append(
                AYVoiceFrame(
                    pitch_semitones=hz_to_semitones(hz) if tone_on else float("nan"),
                    volume_db=0.0 if use_env else level_to_db(level),
                    tone_on=tone_on,
                    noise_on=noise_on,
                    use_envelope=use_env,
                )
            )
        ep = (int(regs[i, 12]) << 8) | int(regs[i, 11])
        r13 = int(regs[i, 13])
        glob = AYGlobalFrame(
            noise_pitch=np_to_noise_pitch(int(regs[i, 6]) & 0x1F),
            env_shape=(r13 & 0x0F) if r13 != ENV_NO_WRITE else 0,
            env_rate=ep_to_env_rate(ep, mclk),
            env_retrigger=r13 != ENV_NO_WRITE,
        )
        out.append(AYStateFrame(voices=(voices[0], voices[1], voices[2]), glob=glob))
    return out
