"""Plan A ``rl`` core — wraps the reverse player :class:`ReversePlayer` as a ``LearnedCore``.

``infer`` runs one non-causal forward pass over the whole feature grid, then **decodes** the raw
head activations into a smooth, finite ``AYState``. It deliberately emits ``AYState`` (never raw
registers): all legality/arbitration stays in the deterministic ``repr.compile`` choke point.

Decoding maps unbounded network outputs into musically sane ranges so an **untrained** network
still produces a legal stream (the A1 canary). Training (A2 warm-start, A4 reward) only changes the
weights, not this contract.

The network is built lazily on first ``infer`` (when the feature dimension is known) and cached.
An optional checkpoint path may be supplied via ``cfg.extra['checkpoint']``.
"""

from __future__ import annotations

import numpy as np

from ...config import RunConfig
from ...repr.state import AYGlobalFrame, AYState, AYStateFrame, AYVoiceFrame, FeatureFrames
from ..base import register_core
from .network import N_VOICES, ReversePlayer

# Decode ranges — bound raw heads to sane musical values (the compiler still clamps to hardware).
_PITCH_CENTER = 60.0      # MIDI C4
_PITCH_SPAN = 30.0        # ±30 semitones ⇒ ~C1½..F#6
_VOL_FLOOR_DB = -60.0
_VOL_CEIL_DB = 0.0
_ENV_RATE_FLOOR_HZ = 0.1

_SILENT = AYVoiceFrame(pitch_semitones=float("nan"), volume_db=float("-inf"), tone_on=False)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic via tanh identity.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class RLCore:
    """Reverse-player core. Deterministic given a fixed checkpoint + seed."""

    def __init__(self, cfg: RunConfig) -> None:
        self._cfg = cfg
        self._net: ReversePlayer | None = None
        self._in_dim: int | None = None
        self._checkpoint = cfg.extra.get("checkpoint")
        self._hidden = int(cfg.extra.get("hidden", 128))

    def _device(self) -> str:
        import torch

        return "cuda" if (self._cfg.use_gpu and torch.cuda.is_available()) else "cpu"

    def _ensure_net(self, in_dim: int) -> None:
        import torch

        if self._net is not None and self._in_dim == in_dim:
            return
        torch.manual_seed(self._cfg.seed)  # reproducible weight init (esp. when untrained)
        net = ReversePlayer(in_dim=in_dim, hidden=self._hidden)
        if self._checkpoint:
            state = torch.load(self._checkpoint, map_location="cpu")
            net.load_state_dict(state.get("model", state))
        net.eval().to(self._device())
        self._net = net
        self._in_dim = in_dim

    def infer(self, feats: FeatureFrames, cfg: RunConfig) -> AYState:
        import torch

        f = np.asarray(feats.feats, dtype=np.float32)
        if f.ndim != 2:
            raise ValueError(f"expected FeatureFrames.feats (T, dim), got {f.shape}")
        n_frames, dim = f.shape
        if n_frames == 0:
            return []
        self._ensure_net(dim)
        assert self._net is not None

        torch.manual_seed(cfg.seed)
        x = torch.from_numpy(f.T).unsqueeze(0).to(self._device())  # (1, dim, T)
        with torch.no_grad():
            out = self._net(x)
        heads = {k: v[0].detach().cpu().numpy() for k, v in out.items()}  # drop batch dim
        return _decode(heads, n_frames)


def _decode(h: dict[str, np.ndarray], n_frames: int) -> AYState:
    """Map raw head arrays → a finite, legal ``AYState`` of length ``n_frames``."""
    pitch = _PITCH_CENTER + _PITCH_SPAN * np.tanh(h["pitch"])                 # (3, T)
    volume = _VOL_FLOOR_DB + (_VOL_CEIL_DB - _VOL_FLOOR_DB) * _sigmoid(h["volume"])
    tone_on = h["tone_logit"] > 0.0
    noise_on = h["noise_logit"] > 0.0
    env_use = h["env_use_logit"] > 0.0
    noise_pitch = _sigmoid(h["noise_pitch"][0])                              # (T,)
    env_rate = _ENV_RATE_FLOOR_HZ + _softplus(h["env_rate"][0])
    env_shape = np.argmax(h["env_shape"], axis=0).astype(int)                # (T,)
    env_retrig = h["env_retrig"][0] > 0.0

    state: AYState = []
    for t in range(n_frames):
        voices = []
        for v in range(N_VOICES):
            audible = bool(tone_on[v, t] or noise_on[v, t])
            if not audible:
                voices.append(_SILENT)
            else:
                voices.append(
                    AYVoiceFrame(
                        pitch_semitones=float(pitch[v, t]),
                        volume_db=float(volume[v, t]),
                        tone_on=bool(tone_on[v, t]),
                        noise_on=bool(noise_on[v, t]),
                        use_envelope=bool(env_use[v, t]),
                    )
                )
        glob = AYGlobalFrame(
            noise_pitch=float(noise_pitch[t]),
            env_shape=int(env_shape[t]),
            env_rate=float(env_rate[t]),
            env_retrigger=bool(env_retrig[t]),
        )
        state.append(AYStateFrame(voices=(voices[0], voices[1], voices[2]), glob=glob))
    return state


@register_core("rl")
def _make_rl(cfg: RunConfig) -> RLCore:
    return RLCore(cfg)
