"""A4 — Regime-1 reward training: head→controls relaxation, perceptual loss, overfit canary.

The §A.11 canary (``test_overfit_one_track``) is the wiring proof: the whole differentiable loop
(features → E → relaxed controls → diff emulator → multi-scale spectral loss) must be able to drive
reconstruction loss down on a single target. The unit tests around it check the relaxation is
differentiable and in-range and that the spectral loss behaves (zero on identical, positive on
different audio).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audio2ay4.chip.diff import DiffAyEmulator, unpack_regs  # noqa: E402
from audio2ay4.config import DEFAULT_FRAME_RATE_HZ, DEFAULT_MASTER_CLOCK_HZ  # noqa: E402
from audio2ay4.models.policy.network import ReversePlayer  # noqa: E402
from audio2ay4.models.policy.relax import controls_from_heads  # noqa: E402
from audio2ay4.repr.compile import EP_MIN, NP_MIN, TP_MIN  # noqa: E402
from audio2ay4.train.reward import (  # noqa: E402
    RewardWeights,
    chroma_loss,
    jitter_penalty,
    multiscale_stft_loss,
    onset_loss,
    reward_loss,
)

MCLK = DEFAULT_MASTER_CLOCK_HZ
FRATE = DEFAULT_FRAME_RATE_HZ
# Small render rate keeps the canary fast; both target and reconstruction use the same emulator.
SR = 8_000


def _tone_regs(tp: int, level: int = 15, n_frames: int = 24) -> np.ndarray:
    """Steady tone on channel A (noise + envelope off)."""
    regs = np.zeros((n_frames, 16), np.uint8)
    regs[:, 0] = tp & 0xFF
    regs[:, 1] = (tp >> 8) & 0x0F
    regs[:, 6] = 1
    regs[:, 7] = 0x3E  # active-low: tone A on
    regs[:, 8] = level & 0x0F
    regs[:, 13] = 0xFF  # no envelope write
    return regs


def _fast_emulator() -> DiffAyEmulator:
    return DiffAyEmulator(render_sr=SR, oversample=1, max_partials=12)


# --------------------------------------------------------------------------------------------- #
# Head → controls relaxation
# --------------------------------------------------------------------------------------------- #

def _dummy_heads(b: int = 1, t: int = 10, *, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    from audio2ay4.models.policy.spec import (
        N_ENV_RATE_BINS,
        N_ENV_SHAPES,
        N_NOISE_LEVELS,
        N_PITCH_BINS,
        N_VOICES,
        N_VOL_LEVELS,
    )

    g = torch.manual_seed(0)
    rg = requires_grad
    return {
        "pitch_logits": torch.randn(b, N_VOICES, N_PITCH_BINS, t, generator=g, requires_grad=rg),
        "volume_logits": torch.randn(b, N_VOICES, N_VOL_LEVELS, t, generator=g, requires_grad=rg),
        "tone_logit": torch.randn(b, N_VOICES, t, generator=g, requires_grad=rg),
        "noise_logit": torch.randn(b, N_VOICES, t, generator=g),
        "env_use_logit": torch.randn(b, N_VOICES, t, generator=g),
        "noise_pitch_logits": torch.randn(b, N_NOISE_LEVELS, t, generator=g),
        "env_rate_logits": torch.randn(b, N_ENV_RATE_BINS, t, generator=g),
        "env_shape": torch.randn(b, N_ENV_SHAPES, t, generator=g),
        "env_retrig": torch.randn(b, 1, t, generator=g),
    }


def test_controls_shapes_and_ranges():
    c = controls_from_heads(_dummy_heads(b=2, t=7), float(MCLK))
    assert c.tone_period.shape == (2, 7, 3)
    assert c.noise_period.shape == (2, 7)
    assert c.env_period.shape == (2, 7)
    assert c.level.shape == (2, 7, 3)
    assert c.env_shape.shape == (2, 7) and c.env_shape.dtype == torch.long
    # Periods / levels are in the hardware-representable ranges the emulator expects.
    assert float(c.tone_period.min()) >= TP_MIN
    assert float(c.noise_period.min()) >= NP_MIN
    assert float(c.env_period.min()) >= EP_MIN
    assert 0.0 <= float(c.level.min()) and float(c.level.max()) <= 15.0
    for gate in (c.tone_gate, c.noise_gate, c.use_env, c.env_retrig):
        assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


def test_controls_select_matches_diffcontrols():
    c = controls_from_heads(_dummy_heads(b=2, t=5), float(MCLK))
    dc = c.select(1)
    assert dc.tone_period.shape == (5, 3)
    assert dc.noise_period.shape == (5,)
    assert torch.allclose(dc.level, c.level[1])


def test_relaxation_is_differentiable():
    heads = _dummy_heads(b=1, t=8, requires_grad=True)
    c = controls_from_heads(heads, float(MCLK))
    # Tone period depends on the pitch logits; level depends on the volume logits.
    c.tone_period.sum().backward(retain_graph=True)
    assert heads["pitch_logits"].grad is not None
    assert torch.isfinite(heads["pitch_logits"].grad).all()
    assert float(heads["pitch_logits"].grad.abs().sum()) > 0.0
    heads["volume_logits"].grad = None
    c.level.sum().backward()
    assert float(heads["volume_logits"].grad.abs().sum()) > 0.0


def test_controls_render_end_to_end_differentiable():
    """The real ReversePlayer → relaxation → emulator path is differentiable end to end."""
    emu = _fast_emulator()
    torch.manual_seed(0)
    dim, t = 16, 12
    net = ReversePlayer(in_dim=dim, hidden=32)
    x = torch.randn(1, dim, t)
    c = controls_from_heads(net(x), float(MCLK))
    audio = emu.render(c.select(0), MCLK, FRATE)
    audio.pow(2).mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients reached the network"
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


# --------------------------------------------------------------------------------------------- #
# Perceptual reward terms
# --------------------------------------------------------------------------------------------- #

def test_multiscale_stft_zero_on_identical():
    x = torch.randn(4000)
    assert float(multiscale_stft_loss(x, x.clone())) == pytest.approx(0.0, abs=1e-6)


def test_multiscale_stft_positive_on_different():
    a = torch.randn(4000)
    b = torch.randn(4000)
    assert float(multiscale_stft_loss(a, b)) > 0.0


def test_multiscale_stft_batched():
    a = torch.randn(3, 4000)
    assert float(multiscale_stft_loss(a, a.clone())) == pytest.approx(0.0, abs=1e-6)


def test_jitter_zero_on_steady_controls():
    heads = _dummy_heads(b=1, t=16)
    # Force every frame identical ⇒ no frame-to-frame change ⇒ zero jitter.
    for k in heads:
        heads[k] = heads[k][..., :1].expand_as(heads[k]).contiguous()
    c = controls_from_heads(heads, float(MCLK))
    assert float(jitter_penalty(c)) == pytest.approx(0.0, abs=1e-6)


def test_reward_loss_parts():
    emu = _fast_emulator()
    heads = _dummy_heads(b=1, t=12)
    c = controls_from_heads(heads, float(MCLK))
    recon = emu.render(c.select(0), MCLK, FRATE).unsqueeze(0)
    total, parts = reward_loss(recon, recon.clone(), c, RewardWeights())
    assert set(parts) == {"spectral", "jitter"}
    assert parts["spectral"] == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------------------------- #
# Chroma (melody) + onset (rhythm) reward terms — the A.4 w3 term, validated against ears
# --------------------------------------------------------------------------------------------- #

def _sine(freq: float, n: int = 12_000) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / SR
    return torch.sin(2.0 * torch.pi * freq * t)


def _square(freq: float, n: int = 12_000) -> torch.Tensor:
    return torch.sign(_sine(freq, n))


def _pulsed(freq: float, period_s: float, n: int = 12_000) -> torch.Tensor:
    """Tone gated on for the first half of each period — gives a clear onset envelope."""
    env = ((torch.arange(n, dtype=torch.float32) / SR) % period_s < period_s / 2).float()
    return _sine(freq, n) * env


def test_chroma_loss_zero_on_identical():
    x = _square(440.0)  # harmonics give real chroma energy
    assert float(chroma_loss(x, x.clone(), SR)) == pytest.approx(0.0, abs=1e-4)


def test_chroma_loss_timbre_invariant():
    """Same note, different timbre must score far better than a tritone away."""
    a = _sine(440.0)
    same_note = _square(440.0)
    tritone = _sine(440.0 * 2.0 ** (6.0 / 12.0))
    assert float(chroma_loss(a, same_note, SR)) < float(chroma_loss(a, tritone, SR))


def test_chroma_loss_octave_equivalent():
    a = _sine(440.0)
    octave = _sine(880.0)
    tritone = _sine(440.0 * 2.0 ** (6.0 / 12.0))
    assert float(chroma_loss(a, octave, SR)) < float(chroma_loss(a, tritone, SR))


def test_chroma_loss_short_input_is_zero():
    short = _sine(440.0, n=1000)  # < n_fft
    assert float(chroma_loss(short, short.clone(), SR)) == 0.0


def test_chroma_loss_differentiable():
    a = _square(440.0).requires_grad_(True)
    b = _square(440.0 * 2.0 ** (3.0 / 12.0))
    loss = chroma_loss(a, b, SR)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert float(a.grad.abs().sum()) > 0.0


def test_onset_loss_low_on_identical():
    x = _pulsed(440.0, period_s=0.1)
    assert float(onset_loss(x, x.clone(), SR)) == pytest.approx(0.0, abs=1e-4)


def test_onset_loss_higher_on_mismatched_rhythm():
    a = _pulsed(440.0, period_s=0.1)
    same = _pulsed(440.0, period_s=0.1)
    off = _pulsed(440.0, period_s=0.037)
    assert float(onset_loss(a, off, SR)) > float(onset_loss(a, same, SR))


def test_onset_loss_differentiable():
    a = _pulsed(440.0, period_s=0.1).requires_grad_(True)
    b = _pulsed(440.0, period_s=0.05)
    loss = onset_loss(a, b, SR)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert float(a.grad.abs().sum()) > 0.0


def test_reward_loss_includes_chroma_onset_when_weighted():
    emu = _fast_emulator()
    heads = _dummy_heads(b=1, t=12)
    c = controls_from_heads(heads, float(MCLK))
    recon = emu.render(c.select(0), MCLK, FRATE).unsqueeze(0)
    weights = RewardWeights(spectral=1.0, jitter=0.0, chroma=1.0, onset=1.0)
    _, parts = reward_loss(recon, recon.clone(), c, weights, sample_rate=SR)
    assert {"spectral", "jitter", "chroma", "onset"} <= set(parts)


# --------------------------------------------------------------------------------------------- #
# §A.11 overfit-one-track canary — the differentiable loop must reach the loss floor
# --------------------------------------------------------------------------------------------- #

def _free_heads(t: int) -> dict[str, torch.nn.Parameter]:
    """Free, learnable head logits (the analysis-by-synthesis variables) for one B=1 track."""
    from audio2ay4.models.policy.spec import (
        N_ENV_RATE_BINS,
        N_ENV_SHAPES,
        N_NOISE_LEVELS,
        N_PITCH_BINS,
        N_VOICES,
        N_VOL_LEVELS,
    )

    g = torch.manual_seed(0)

    def p(*shape: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(0.1 * torch.randn(*shape, generator=g))

    return {
        "pitch_logits": p(1, N_VOICES, N_PITCH_BINS, t),
        "volume_logits": p(1, N_VOICES, N_VOL_LEVELS, t),
        "tone_logit": p(1, N_VOICES, t),
        "noise_logit": p(1, N_VOICES, t),
        "env_use_logit": p(1, N_VOICES, t),
        "noise_pitch_logits": p(1, N_NOISE_LEVELS, t),
        "env_rate_logits": p(1, N_ENV_RATE_BINS, t),
        "env_shape": p(1, N_ENV_SHAPES, t),
        "env_retrig": p(1, 1, t),
    }


def test_overfit_one_track():
    """Optimising the relaxed controls of one track must drive reconstruction loss to its floor.

    This is the §A.11 wiring canary: features→E→controls→emulator→loss is differentiable end to end
    (proven separately in ``test_controls_render_end_to_end_differentiable``); here we confirm the
    relaxation + emulator + multi-scale loss can actually be *minimised* down to a near-zero floor.

    Multi-scale spectral loss famously carries almost no gradient to *slide* a tone's fundamental
    (two non-overlapping pure tones look equidistant) — pitch must come from the A2 warm-start, and
    Regime-1 only ever fine-tunes a warm-started net. We therefore seed the pitch at the target
    (the warm-start's job) and let the loop recover the well-conditioned freedoms: amplitude, voice
    gating (silence B/C), and turning noise/envelope off.
    """
    from audio2ay4.models.policy.spec import pitch_to_bin
    from audio2ay4.repr.compile import hz_to_semitones

    torch.manual_seed(0)
    emu = _fast_emulator()
    n_frames = 24

    # Target audio: a steady mid tone rendered by the same differentiable emulator.
    regs = _tone_regs(tp=300, level=15, n_frames=n_frames)
    with torch.no_grad():
        target = emu.render(unpack_regs(regs), MCLK, FRATE).unsqueeze(0)

    heads = _free_heads(n_frames)
    # Seed every voice's pitch at the target bin (the warm-start prior); the loop must still learn
    # which voice is audible and at what level — those have well-behaved gradients.
    target_bin = pitch_to_bin(hz_to_semitones(MCLK / (16.0 * 300)))
    with torch.no_grad():
        heads["pitch_logits"][:, :, target_bin, :] += 6.0

    opt = torch.optim.Adam(list(heads.values()), lr=0.05)

    def _step() -> float:
        c = controls_from_heads(heads, float(MCLK), tau=0.5)
        recon = emu.render(c.select(0), MCLK, FRATE).unsqueeze(0)
        loss = multiscale_stft_loss(recon, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())

    initial = _step()
    final = initial
    for _ in range(300):
        final = _step()

    assert final < 0.1 * initial, f"reconstruction loss did not fall: {initial:.4f} → {final:.4f}"


# --------------------------------------------------------------------------------------------- #
# A5 — augmented (domain-gap) reward forward pass
# --------------------------------------------------------------------------------------------- #

def test_reward_forward_augment_runs_and_backprops():
    """A5 augment mode: input = SUNO-degraded re-render of the reference, reward vs the clean target.

    The whole loop must still be differentiable end to end (grads reach the net) even though the
    encoder input is now re-extracted from coloured audio rather than the cached clean features.
    """
    from audio2ay4.config import RunConfig
    from audio2ay4.train.reward_train import reward_forward

    run = RunConfig(sample_rate=SR, frame_rate_hz=FRATE, feat_kind="mel", use_gpu=False)
    emu = _fast_emulator()
    regs = _tone_regs(tp=300, level=15, n_frames=24)
    feats = np.zeros((regs.shape[0], 80), np.float32)        # unused in augment mode (replaced)
    sample = (feats, regs, MCLK, FRATE)

    net = ReversePlayer(in_dim=80, hidden=32)
    total, parts = reward_forward(
        net, emu, [sample], device="cpu", weights=RewardWeights(), tau=1.0,
        run=run, augment=True, aug_strength=1.0, rng=np.random.default_rng(0),
    )

    assert torch.isfinite(total)
    assert total.requires_grad
    assert "spectral" in parts and "jitter" in parts
    total.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients reached the network"
    assert any(torch.any(g != 0) for g in grads), "all gradients were zero"
