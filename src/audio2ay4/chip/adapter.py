"""Adapter to audio2ay3's proven chip core (emulator + YM read/write).

The heavy dependency (audio2ay3, which pulls numba for the emulator) is imported **lazily** inside
each function so that importing :mod:`audio2ay4` stays light. If audio2ay3 is missing, a clear
install hint is raised.
"""

from __future__ import annotations

import numpy as np

from ..repr.state import YmSong

_INSTALL_HINT = (
    "audio2ay4's chip core reuses audio2ay3's proven AY emulator + YM I/O, which is not installed.\n"
    "Install it with one of:\n"
    "    pip install -e .[ay3]                      # from git\n"
    "    pip install -e ../audio2ay3                # against a local checkout\n"
    "See https://github.com/parallelno/audio2ay3"
)


def _require_ay3():
    try:
        import audio2ay3  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from exc


def _to_ay3_song(song: YmSong):
    """Convert an audio2ay4 :class:`YmSong` into an audio2ay3 ``ymformat.model.YmSong``."""
    _require_ay3()
    from audio2ay3.ymformat.model import YmSong as Ay3Song

    return Ay3Song(
        frames=np.ascontiguousarray(song.regs, dtype=np.uint8),
        master_clock=int(song.master_clock_hz),
        frame_rate=int(song.frame_rate_hz),
        loop_frame=int(song.loop_frame or 0),
        n_chips=1,
    )


def render_song(song: YmSong, sample_rate: int = 44_100) -> np.ndarray:
    """Render a :class:`YmSong` to mono float32 PCM via audio2ay3's emulator (ground truth)."""
    _require_ay3()
    from audio2ay3.chip import Ay3Emulator

    emu = Ay3Emulator(render_sr=int(sample_rate))
    return emu.render_song(_to_ay3_song(song))


def write_ym(song: YmSong, path: str, version: str = "YM6") -> None:
    """Serialise a :class:`YmSong` to ``path`` using audio2ay3's YM writer."""
    from audio2ay3.ymformat import ym_writer  # lazy; _require_ay3 via _to_ay3_song

    ym_writer.write(_to_ay3_song(song), path, version=version)


def read_ym(path: str) -> YmSong:
    """Read a YM file (YM2/3/5/6, transparent LHA depack) into an audio2ay4 :class:`YmSong`."""
    _require_ay3()
    from audio2ay3.ymformat import ym_reader

    with open(path, "rb") as fh:
        s = ym_reader.from_bytes(fh.read())
    regs = np.ascontiguousarray(s.frames, dtype=np.uint8)
    if regs.shape[1] > 16:  # keep chip 0 for the single-chip contract
        regs = regs[:, :16]
    return YmSong(
        regs=regs,
        master_clock_hz=int(s.master_clock),
        frame_rate_hz=int(s.frame_rate),
        loop_frame=int(getattr(s, "loop_frame", 0)),
        meta={"name": s.name, "author": s.author, "comment": s.comment},
    )
