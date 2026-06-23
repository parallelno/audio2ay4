"""A3 — differentiable AY emulator (`chip/diff`) validated against the trusted `audio2ay3` oracle.

The twin trades bit-exactness for differentiability, so it is validated **spectrally / by energy**
(dominant pitch, RMS, broadband noise) rather than sample-wise, plus a gradient-flow check that is
the whole point of building it. Oracle comparisons skip cleanly if `audio2ay3` is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audio2ay4.chip.diff import AY_DAC, DiffAyEmulator, DiffControls, unpack_regs  # noqa: E402
from audio2ay4.config import DEFAULT_FRAME_RATE_HZ, DEFAULT_MASTER_CLOCK_HZ  # noqa: E402

MCLK = DEFAULT_MASTER_CLOCK_HZ
FRATE = DEFAULT_FRAME_RATE_HZ
SR = 44_100


def _tone_regs(tp: int, level: int = 15, n_frames: int = 25) -> np.ndarray:
    """Register stream: a steady tone on channel A, noise + envelope off."""
    regs = np.zeros((n_frames, 16), np.uint8)
    regs[:, 0] = tp & 0xFF
    regs[:, 1] = (tp >> 8) & 0x0F
    regs[:, 6] = 1
    regs[:, 7] = 0x3E  # active-low: tone A enabled (bit0=0), everything else off
    regs[:, 8] = level & 0x0F
    regs[:, 13] = 0xFF  # no envelope write
    return regs


def _noise_regs(npv: int, level: int = 15, n_frames: int = 25) -> np.ndarray:
    """Register stream: steady noise on channel A, tone + envelope off."""
    regs = np.zeros((n_frames, 16), np.uint8)
    regs[:, 6] = npv & 0x1F
    regs[:, 7] = 0x37  # active-low: noise A enabled (bit3=0), tones off
    regs[:, 8] = level & 0x0F
    regs[:, 13] = 0xFF
    return regs


def _dominant_hz(pcm: np.ndarray, sr: int = SR) -> float:
    x = np.asarray(pcm, np.float64)
    x = x - x.mean()
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(freqs[np.argmax(mag)])


def _rms(pcm) -> float:
    a = pcm.detach().cpu().numpy() if hasattr(pcm, "detach") else np.asarray(pcm)
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))


def _oracle():
    ay3 = pytest.importorskip("audio2ay3")  # noqa: F841
    from audio2ay3.chip import Ay3Emulator

    return Ay3Emulator(render_sr=SR)


# --------------------------------------------------------------------------------------------- #
# Self-contained checks (no oracle needed)
# --------------------------------------------------------------------------------------------- #

def test_dac_table_mirrors_audio2ay3():
    ay3 = pytest.importorskip("audio2ay3")  # noqa: F841
    from audio2ay3.chip.volume_tables import AY_DAC as ORACLE_DAC

    assert np.allclose(np.array(AY_DAC), np.asarray(ORACLE_DAC))


def test_render_empty_is_empty():
    emu = DiffAyEmulator()
    out = emu.render_regs(np.zeros((0, 16), np.uint8), MCLK, FRATE)
    assert out.shape == (0,)


def test_tone_dominant_frequency_matches_formula():
    tp = 200
    emu = DiffAyEmulator()
    pcm = emu.render_regs(_tone_regs(tp), MCLK, FRATE).detach().cpu().numpy()
    expected = MCLK / (16.0 * tp)
    assert abs(_dominant_hz(pcm) - expected) < 5.0


def test_louder_level_has_more_energy():
    emu = DiffAyEmulator()
    quiet = _rms(emu.render_regs(_tone_regs(300, level=6), MCLK, FRATE))
    loud = _rms(emu.render_regs(_tone_regs(300, level=15), MCLK, FRATE))
    assert loud > quiet * 2.0


def test_render_is_differentiable_wrt_controls():
    n = 20
    tone_period = torch.full((n, 3), 300.0, requires_grad=True)
    ones = torch.ones(n, 3)
    controls = DiffControls(
        tone_period=tone_period,
        noise_period=torch.ones(n),
        env_period=torch.ones(n),
        tone_gate=torch.cat([ones[:, :1], torch.zeros(n, 2)], dim=1),
        noise_gate=torch.zeros(n, 3),
        level=torch.full((n, 3), 15.0),
        use_env=torch.zeros(n, 3),
        env_shape=torch.zeros(n, dtype=torch.long),
        env_retrig=torch.zeros(n),
    )
    emu = DiffAyEmulator()
    pcm = emu.render(controls, MCLK, FRATE)
    pcm.pow(2).mean().backward()
    g = tone_period.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g[:, 0].abs().sum() > 0  # gradient reaches the audible tone channel


def test_unpack_regs_roundtrips_key_fields():
    regs = _tone_regs(300, level=11, n_frames=4)
    regs[:, 6] = 7  # noise period
    c = unpack_regs(regs)
    assert torch.allclose(c.tone_period[:, 0], torch.full((4,), 300.0))
    assert torch.allclose(c.noise_period, torch.full((4,), 7.0))
    assert torch.allclose(c.tone_gate[:, 0], torch.ones(4))
    assert torch.allclose(c.tone_gate[:, 1], torch.zeros(4))
    assert torch.allclose(c.level[:, 0], torch.full((4,), 11.0))


# --------------------------------------------------------------------------------------------- #
# Oracle (audio2ay3) comparisons — spectral / energy, not sample-exact
# --------------------------------------------------------------------------------------------- #

def test_length_matches_oracle():
    emu = DiffAyEmulator()
    regs = _tone_regs(300, n_frames=30)
    ours = emu.render_regs(regs, MCLK, FRATE)
    ref = _oracle().render_frames(regs, MCLK, FRATE)
    assert abs(ours.shape[0] - ref.shape[0]) <= 1


@pytest.mark.parametrize("tp", [120, 300, 700])
def test_tone_pitch_agrees_with_oracle(tp):
    emu = DiffAyEmulator()
    regs = _tone_regs(tp, n_frames=40)
    ours = emu.render_regs(regs, MCLK, FRATE).detach().cpu().numpy()
    ref = _oracle().render_frames(regs, MCLK, FRATE)
    assert abs(_dominant_hz(ours) - _dominant_hz(ref)) < 5.0


def test_tone_rms_in_oracle_ballpark():
    emu = DiffAyEmulator()
    regs = _tone_regs(300, n_frames=40)
    ours = _rms(emu.render_regs(regs, MCLK, FRATE))
    ref = _rms(_oracle().render_frames(regs, MCLK, FRATE))
    assert 0.5 < ours / ref < 2.0


def test_noise_is_broadband_like_oracle():
    emu = DiffAyEmulator()
    regs = _noise_regs(8, n_frames=40)
    ours = emu.render_regs(regs, MCLK, FRATE).detach().cpu().numpy()
    ref = _oracle().render_frames(regs, MCLK, FRATE)
    # Noise has no single dominant partial: spectral flatness (geo/arith mean) is well above a tone's.
    def flatness(x):
        x = np.asarray(x, np.float64)
        p = np.abs(np.fft.rfft(x - x.mean())) ** 2 + 1e-12
        return float(np.exp(np.mean(np.log(p))) / np.mean(p))

    assert flatness(ours) > 0.05
    assert 0.3 < _rms(emu.render_regs(regs, MCLK, FRATE)) / _rms(ref) < 3.0
