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
from dataclasses import replace

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


def _cmd_train(args: argparse.Namespace) -> int:
    if args.plan != "rl":
        return _cmd_stub("train", "plan-b-diffusion.md")
    from .config import TrainConfig
    from .data.corpus import find_ym_files
    from .train.warmstart import train_warmstart

    ym_paths = find_ym_files(args.corpus)
    if not ym_paths:
        print(f"No .ym files found under {args.corpus}.", file=sys.stderr)
        return 1
    if args.limit > 0:
        ym_paths = ym_paths[: args.limit]

    run = _cfg(args)
    run = replace(run, core="rl", extra={**run.extra, **({"checkpoint": args.out} if args.out else {})})
    train_cfg = TrainConfig(
        plan="rl", run=run, batch_size=args.batch_size, lr=args.lr,
        max_steps=args.max_steps, corpus_dir=args.corpus, cache_dir=args.cache_dir,
    )
    ckpt = train_warmstart(train_cfg, ym_paths)
    print(f"Checkpoint: {ckpt}")
    return 0


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

    t = sub.add_parser("train", help="train a learned core (Plan A warm-start)")
    t.add_argument("plan", nargs="?", default="rl", choices=["rl", "diffusion"], help="which plan")
    t.add_argument("--corpus", default="corpus/ym", help="directory of .ym training tunes")
    t.add_argument("--out", default="", help="checkpoint output path (default: <cache>/warmstart_rl.pt)")
    t.add_argument("--cache-dir", default=".cache", help="feature/pair cache directory")
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--max-steps", type=int, default=2000)
    t.add_argument("--limit", type=int, default=0, help="use only the first N tunes (0 = all)")
    _add_common(t)
    t.set_defaults(func=_cmd_train)

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
