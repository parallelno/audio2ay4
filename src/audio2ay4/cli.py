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
    p.add_argument("--checkpoint", default="",
                   help="trained core checkpoint, e.g. a warm-start .pt (used by --core rl)")
    p.add_argument("--hidden", type=int, default=128,
                   help="hidden width of the learned core (must match the checkpoint)")


def _cfg(args: argparse.Namespace) -> RunConfig:
    extra: dict = {}
    if getattr(args, "checkpoint", ""):
        extra["checkpoint"] = args.checkpoint
    if getattr(args, "hidden", None):
        extra["hidden"] = args.hidden
    return RunConfig(
        core=args.core,
        master_clock_hz=args.clock,
        frame_rate_hz=args.frame_rate,
        sample_rate=args.sample_rate,
        extra=extra,
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
    print(f"{'track':<32} {'spec_dist':>10} {'chroma':>8} {'onset':>8} {'stability':>10} {'legal':>6} {'frames':>8}")
    for r in results:
        print(f"{r.name[:32]:<32} {r.spectral_distance:>10.4f} {r.chroma_sim:>8.4f} "
              f"{r.onset_sim:>8.4f} {r.stability:>10.4f} "
              f"{('yes' if r.legal else 'NO'):>6} {r.n_frames:>8}")
    agg = aggregate(results)
    print(f"\nmean over {int(agg['count'])}: spec_dist={agg['spectral_distance']:.4f} "
          f"chroma={agg['chroma_sim']:.4f} onset={agg['onset_sim']:.4f} "
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

    ym_paths = find_ym_files(args.corpus)
    if not ym_paths:
        print(f"No .ym files found under {args.corpus}.", file=sys.stderr)
        return 1
    if args.limit > 0:
        ym_paths = ym_paths[: args.limit]

    if args.regime == "reward":
        return _train_reward(args, ym_paths)

    from .train.warmstart import train_warmstart

    run = _cfg(args)
    reg = {"weight_decay": args.weight_decay, "dropout": args.dropout, "feat_noise": args.feat_noise}
    run = replace(run, core="rl", extra={
        **run.extra, **reg, **({"checkpoint": args.out} if args.out else {})
    })
    train_cfg = TrainConfig(
        plan="rl", run=run, batch_size=args.batch_size, lr=args.lr,
        max_steps=args.max_steps, corpus_dir=args.corpus, cache_dir=args.cache_dir,
    )
    ckpt = train_warmstart(train_cfg, ym_paths, workers=args.workers,
                           window=(args.window or None), val_frac=args.val_frac)
    print(f"Checkpoint: {ckpt}")
    return 0


def _train_reward(args: argparse.Namespace, ym_paths: list[str]) -> int:
    """Regime-1 reward (analysis-by-synthesis) fine-tuning — design A.4."""
    from .config import TrainConfig
    from .train.reward import RewardWeights
    from .train.reward_train import train_reward

    run = _cfg(args)
    run = replace(run, core="rl", extra={
        **run.extra, **({"checkpoint": args.out} if args.out else {})
    })
    train_cfg = TrainConfig(
        plan="rl", run=run, batch_size=args.batch_size, lr=args.lr,
        max_steps=args.max_steps, corpus_dir=args.corpus, cache_dir=args.cache_dir,
    )
    weights = RewardWeights(spectral=args.w_spec, jitter=args.w_jitter,
                            chroma=args.w_chroma, onset=args.w_onset)
    ckpt = train_reward(
        train_cfg, ym_paths,
        init_checkpoint=(args.init or None),
        workers=args.workers,
        window=(args.window or None),
        weights=weights,
        tau=args.tau,
        augment=args.augment,
        aug_strength=args.augment_strength,
    )
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

    t = sub.add_parser("train", help="train a learned core (Plan A warm-start / reward)")
    t.add_argument("plan", nargs="?", default="rl", choices=["rl", "diffusion"], help="which plan")
    t.add_argument("--regime", default="warmstart", choices=["warmstart", "reward"],
                   help="warmstart = supervised A2; reward = Regime-1 analysis-by-synthesis (A4)")
    t.add_argument("--corpus", default="corpus/ym", help="directory of .ym training tunes")
    t.add_argument("--out", default="", help="checkpoint output path (default: <cache>/warmstart_rl.pt)")
    t.add_argument("--cache-dir", default=".cache", help="feature/pair cache directory")
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--max-steps", type=int, default=2000)
    t.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of tunes held out for validation loss (0 = none)")
    t.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay (L2 regularization; 0 = off)")
    t.add_argument("--dropout", type=float, default=0.1,
                   help="dropout in the TCN residual blocks (0 = off)")
    t.add_argument("--feat-noise", type=float, default=0.0,
                   help="train-time Gaussian feature jitter, in units of the feature std (0 = off)")
    t.add_argument("--window", type=int, default=512,
                   help="train on random N-frame windows (0 = whole songs; slower)")
    t.add_argument("--workers", type=int, default=0,
                   help="parallel render processes (0 = all CPU cores)")
    t.add_argument("--limit", type=int, default=0, help="use only the first N tunes (0 = all)")
    # Reward-regime (A4) options.
    t.add_argument("--init", default="",
                   help="[reward] warm-start checkpoint to initialise E from (recommended)")
    t.add_argument("--w-spec", type=float, default=1.0,
                   help="[reward] multi-scale spectral term weight")
    t.add_argument("--w-jitter", type=float, default=0.02,
                   help="[reward] control jitter (stability) penalty weight")
    t.add_argument("--w-chroma", type=float, default=0.0,
                   help="[reward] chroma (melody/harmony) loss weight — the headline term")
    t.add_argument("--w-onset", type=float, default=0.0,
                   help="[reward] onset (rhythm/timing) loss weight")
    t.add_argument("--tau", type=float, default=1.0,
                   help="[reward] soft-argmax temperature for the head relaxation")
    t.add_argument("--augment", action="store_true",
                   help="[reward] A5: train on SUNO-style degraded input audio (domain-gap bridge)")
    t.add_argument("--augment-strength", type=float, default=1.0,
                   help="[reward] augmentation intensity in [0,1] (only with --augment)")
    _add_common(t)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("eval", help="convert audio (file or dir) and score fidelity/stability/legality")
    e.add_argument("input", help="audio file or directory of audio files")
    _add_common(e)
    e.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page (cp1252) that can't encode characters like
    # "→"/"…" in our progress output, which would raise UnicodeEncodeError mid-run. Emit UTF-8 and
    # replace anything the stream truly can't represent so logging never crashes the training run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
