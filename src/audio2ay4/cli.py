"""Command-line interface: ``audio2ay4 <command> ...`` (entry point in pyproject scripts).

Commands:
    convert   audio → YM
    preview   YM → audio, or audio → YM → audio (round-trip)
    validate  assert a register stream is hardware-legal
    train     (Plan A / Plan B) — stub, see design plans
    eval      (metrics) — stub, see design plans
"""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_FRAME_RATE_HZ, DEFAULT_MASTER_CLOCK_HZ, RunConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--core", default="dummy", help="learned core name (default: dummy)")
    p.add_argument("--clock", type=int, default=DEFAULT_MASTER_CLOCK_HZ, help="AY master clock (Hz)")
    p.add_argument("--frame-rate", type=int, default=DEFAULT_FRAME_RATE_HZ, help="frames per second")
    p.add_argument("--sample-rate", type=int, default=44_100, help="audio sample rate (Hz)")


def _cfg(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        core=args.core,
        master_clock_hz=args.clock,
        frame_rate_hz=args.frame_rate,
        sample_rate=args.sample_rate,
    )


def _cmd_convert(args: argparse.Namespace) -> int:
    from .convert import convert_audio_to_ym

    song = convert_audio_to_ym(args.input, args.output, _cfg(args))
    print(f"Wrote {args.output} ({song.n_frames} frames @ {song.frame_rate_hz} Hz).")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from .preview import preview_audio, preview_ym

    cfg = _cfg(args)
    if args.input.lower().endswith(".ym"):
        preview_ym(args.input, args.output, cfg)
    else:
        preview_audio(args.input, args.output, cfg)
    print(f"Wrote {args.output}.")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from . import chip
    from .chip.legality import is_legal

    song = chip.read_ym(args.input)
    ok = is_legal(song.regs)
    print(f"{args.input}: {'LEGAL' if ok else 'ILLEGAL'} ({song.n_frames} frames).")
    return 0 if ok else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval import aggregate, evaluate_path

    results = evaluate_path(args.input, _cfg(args))
    if not results:
        print(f"No audio files found at {args.input}.", file=sys.stderr)
        return 1
    print(f"{'track':<32} {'spec_dist':>10} {'stability':>10} {'legal':>6} {'frames':>8}")
    for r in results:
        print(f"{r.name[:32]:<32} {r.spectral_distance:>10.4f} {r.stability:>10.4f} "
              f"{('yes' if r.legal else 'NO'):>6} {r.n_frames:>8}")
    agg = aggregate(results)
    print(f"\nmean over {int(agg['count'])}: spec_dist={agg['spectral_distance']:.4f} "
          f"stability={agg['stability']:.4f} legality_rate={agg['legality_rate']:.3f}")
    return 0 if agg["legality_rate"] >= 1.0 else 1


def _cmd_stub(name: str, plan: str) -> int:
    print(f"`{name}` is not implemented yet — see design/{plan}.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio2ay4", description="Audio → AY (YM) converter & preview.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="audio → YM")
    c.add_argument("input")
    c.add_argument("output")
    _add_common(c)
    c.set_defaults(func=_cmd_convert)

    p = sub.add_parser("preview", help="YM → audio, or audio → YM → audio")
    p.add_argument("input")
    p.add_argument("output")
    _add_common(p)
    p.set_defaults(func=_cmd_preview)

    v = sub.add_parser("validate", help="check a YM register stream is hardware-legal")
    v.add_argument("input")
    v.set_defaults(func=_cmd_validate)

    t = sub.add_parser("train", help="(stub) train a learned core")
    t.set_defaults(func=lambda _a: _cmd_stub("train", "plan-a-reinforcement-learning.md / plan-b-diffusion.md"))

    e = sub.add_parser("eval", help="convert audio (file or dir) and score fidelity/stability/legality")
    e.add_argument("input", help="audio file or directory of audio files")
    _add_common(e)
    e.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
