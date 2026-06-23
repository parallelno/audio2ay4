# audio2ay4 — repo facts (verified )

> Snapshot of the assistant's **repository memory** for this project, exported so it travels with
> the git checkout. On the new machine, paste this back into repo memory (see
> [README.md](README.md) in this folder) so the assistant resumes with full context.

## What it is
Audio (mp3/wav/...) → AY-3-8910/YM2149 register stream (.ym) + audio preview.

## INPUT DOMAIN (important constraint)
- Input is NOT real instruments. Input = SUNO-generated **synthetic chiptune-imitation** audio (AI music that mimics chiptune style, rendered as ordinary full-band/stereo/lossy audio).
- Consequence: narrow domain gap vs YM-rendered training audio (much closer than real instruments) → Plan A self-supervised reward fits well.
- Suno may exceed AY budget (>3 voices, drums, effects) → model must ARRANGE/reduce down to 3 tone + 1 noise + 1 env; perfect reproduction impossible.
- Reward/metric should weight musical structure (pitch/onset/chroma) over exact timbre (we translate Suno-timbre → AY-timbre). Spectral-magnitude match has an irreducible floor.
- Augmentation in data/ should mimic Suno coloration (stereo→mono, reverb, EQ, lossy MP3 codec, limiting) to bridge YM-render → Suno gap.
- Real-task eval set = a few SUNO chiptune clips (NOT real instruments). Two ML plans in `design/`: Plan A = RL reverse player, Plan B = conditional diffusion. Milestone 0 skeleton implemented.

## Environment / commands (Windows, pwsh)
- Default `python` is 2.7 — DO NOT use. Use `py -3.11` (3.11.0 available).
- Venv: `.venv` at repo root. Run tools via `.\.venv\Scripts\python.exe` / `.\.venv\Scripts\audio2ay4.exe`.
- Install: `py -3.11 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"`
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` (8 tests, all pass with audio2ay3 installed; 1 skips without it).

## audio2ay3 reuse (the proven emulator — do NOT reinvent)
- Local checkout: `d:\Work\Programming\audio2ay3`. Install editable: `pip install -e d:\Work\Programming\audio2ay3`.
- Reference ONLY audio2ay3 (never audio2ay / audio2ay2).
- Verified API: `audio2ay3.chip.Ay3Emulator(chip=None, render_sr=44100, oversample=2, dac=None)` → `.render_song(song)->np.ndarray` mono float32.
- `audio2ay3.ymformat.model.YmSong(frames, master_clock, frame_rate, loop_frame, version, name, author, comment, n_chips)`.
- `audio2ay3.ymformat.ym_writer.write(song, path, version='YM6')`; `ym_reader.from_bytes(data)->YmSong`.
- Bridged in `src/audio2ay4/chip/adapter.py` (lazy import). audio2ay4 has its OWN `YmSong` (regs/master_clock_hz/frame_rate_hz) — adapter converts at boundary.

## Architecture choke points
- Register compiler `repr/compile.py::compile_state` is the SINGLE deterministic legality gate. Learned cores emit smooth `AYState` only; never raw registers.
- `chip/legality.py::is_legal` is pure (numpy only, no audio2ay3) — test oracle.
- `models/base.py` registry + `LearnedCore` Protocol(`infer(feats,cfg)->AYState`). `get_core("rl"|"diffusion")` raises NotImplementedError pointing to design plans. `dummy` core = placeholder baseline.
- Pipelines: `convert/pipeline.py` (audio→ym), `preview/render.py` (ym→audio). CLI: `cli.py` subcommands convert/preview/validate (+train/eval stubs).
- Core stays numpy-only importable; torch/soundfile/audio2ay3 are optional extras (audio/neural/ay3), imported lazily.

## Milestone 0 — COMPLETE (design §7 all items green)
- data/: `scan_corpus(root)` (dedup by sha1, lazy audio2ay3 read), `split_by_tune(entries,ratios,seed)` (pure, stable hash, no leakage), `build_pair(ym_path,cfg,cache_dir=None)->TrainingPair` (render via emulator → features + ground-truth regs target; npz disk cache keyed by file sha1 + cfg; aligns feats/regs to min length). TrainingPair.target_state = parse_song(regs) lazily.
- eval/: `spectral_distance(a,b)` multi-res STFT log-mag L1 (numpy), `stability(regs)` frame-change rate 0..1, `legality_rate(list)`, `evaluate_audio/evaluate_path/aggregate` harness (convert→render→compare to input). CLI `eval <file|dir>` prints per-track + mean table; exit 1 if legality_rate<1.
- Feature caching satisfied via data/pairing npz cache.

## Test/lint status
- `.\.venv\Scripts\python.exe -m pytest -q` → 19 passed (audio2ay3 installed). Pure tests (compiler/legality/splits/metrics) run without audio2ay3; pairing+convert tests use `pytest.importorskip("audio2ay3")`.
- `.\.venv\Scripts\python.exe -m ruff check src tests` → clean.
- Linter (get_errors) flags "Unable to import numpy/audio2ay3" — IGNORE, that's the editor's linter env lacking the venv; code runs fine in .venv.

## Verified end-to-end
1s tone.wav → 50-frame YM @50Hz → 1s preview.wav, validate=LEGAL. `eval` on a tone → spec_dist/stability/legal table. Works.

## HARDWARE CONSTRAINT (this machine)
- This PC has a BROKEN CPU → native code (audio2ay3 LHA depack / emulator render) can hard-segfault (exit 1, no Python traceback). NOT a data-corruption bug. IGNORE such crashes locally.
- Do NOT validate/render the corpus by reading .ym here (e.g. scan_corpus over all files crashes mid-scan). All heavy training/testing runs on a SECOND machine.
- Example: `corpus/ym/- unknown/39kshock.ym` is a VALID LHA (header `\x1f\xed-lh5-`, contains 39KBS.BIN) yet crashes read_ym here — CPU fault, file is fine.

## Corpus — DOWNLOADED (Modland YM)
- Downloader: `scripts/download_modland_ym.py` (stdlib-only, threaded, resume/skip-existing, atomic .part, retries+backoff, failures→corpus/ym/_failures.log). Default list = `scripts/modland_ym.txt` (resolved next to script).
- List `scripts/modland_ym.txt`: 4961 relative paths `YM/<author>/<file>.ym`, 156 authors. URL = `https://modland.com/pub/modules/` + urllib.parse.quote(path, safe="/").
- Downloaded ALL 4961 .ym → `corpus/ym/<author>/<file>.ym` (leading `YM/` stripped), 17.3 MB, 0 failures.
- Mixed clocks: Atari ST tunes = 2_000_000 Hz, ZX = 1.7734 MHz. Per-song master_clock_hz handles it.
- Re-run downloader anytime to resume/retry (skips existing non-empty files).

## Git tracking policy (IMPORTANT — data IS committed)
- The .ym corpus, samples, and the emulator tool are COMMITTED to git (training runs on a 2nd machine that needs them via the repo).
- `.gitignore` no longer has global `*.ym/*.wav/*.mp3` ignores. Tracked: `corpus/` (4961 ym), `samples/`, `tools/`. Ignored junk only: `corpus/**/*.part`, `corpus/**/_failures.log`, `**/_scanprobe.log`, `.cache/`.
- `tools/ay_emul/` = Windows AY emulator (Ay_Emul.exe + bass*.dll, 8.1 MB, 77 files incl .ays/.chm/.po) — for users to LISTEN to .ym output and validate results. Not used by code.
- `samples/`: `long/` = 6 Suno-style chiptune mp3s (real-task conversion inputs), `short/` = 4 wav + trumpet.ogg test clips, `ym/song01.ym` = reference fixture.

## Plan A — A1 DONE (reverse player E skeleton)
- torch installed in .venv: `torch 2.12.1+cpu` (CPU wheel from download.pytorch.org/whl/cpu). torchaudio NOT installed (E doesn't need it; feats come in as numpy).
- `models/policy/network.py`: `ReversePlayer(nn.Module)` — non-causal dilated TCN (GroupNorm+GELU residual blocks, symmetric padding → length-preserving, any T). in (B,in_dim,T)→dict of per-frame heads: pitch/volume/tone_logit/noise_logit/env_use_logit (each B,3,T), noise_pitch/env_rate/env_retrig (B,1,T), env_shape (B,16,T). Consts N_VOICES=3, N_ENV_SHAPES=16.
- `models/policy/core.py`: `RLCore` implements LearnedCore.infer(feats,cfg)->AYState. Lazy-builds net on first infer (in_dim=feats.shape[1]); seeds torch.manual_seed(cfg.seed) BEFORE constructing net so untrained init is reproducible. Optional checkpoint via cfg.extra['checkpoint'] (loads state.get('model',state)); hidden via cfg.extra['hidden'] (default 128). `_decode` maps raw heads→finite legal AYState: pitch=60±30*tanh, volume=[-60,0] via sigmoid, gates=logit>0, noise_pitch=sigmoid, env_rate=0.1+softplus, env_shape=argmax. Voice silent (NaN/-inf) only when NOT(tone_on or noise_on). Registered via `@register_core("rl")`.
- `models/policy/__init__.py` imports core (registers 'rl'). `models/base.py::get_core` now lazy-imports `.policy` for name=="rl" (ImportError→NotImplementedError telling to `pip install -e ".[neural]"`). diffusion still NotImplementedError.
- Test `tests/test_policy_rl.py` (pytest.importorskip torch): rl→AYState→compile_state legal, determinism (same seed→same regs), empty input→[], head shapes. CPU-only, no emulator render (honors broken-CPU constraint).
- Status: 23 tests pass (was 19), ruff clean. Did NOT run native convert/render here (CPU segfault risk).

## Plan A — A2 DONE (supervised warm-start)
- `models/policy/spec.py` (torch-free): single source for head spec — N_VOICES=3, N_ENV_SHAPES=16, decode ranges PITCH_CENTER=60/PITCH_SPAN=30, VOL_FLOOR_DB=-60/VOL_CEIL_DB=0, ENV_RATE_FLOOR_HZ=0.1. network.py+core.py refactored to import from spec (removed their private dups).
- `train/targets.py` (torch-free): `build_targets(regs, master_clock_hz, frame_rate_hz)->dict` via parse_song (single inverse). Arrays: pitch/volume/tone/noise/env_use (3,T), noise_pitch/env_rate/env_retrig (T,), env_shape (T,)int64. Continuous clamped to head ranges; env_rate≥floor (loss logs it).
- `train/warmstart_loss.py` (torch): `WarmstartWeights` dataclass + `warmstart_loss(heads,targets,pad_mask,weights)->(total,parts)`. Decodes continuous heads with SAME transforms as inference (tanh/sigmoid/softplus). Masked: pitch where tone_on, volume where audible&~env_use, env_rate/env_shape only where env_retrig. Gates=BCEWithLogits, env_shape=CE, continuous=MSE (env_rate in log space).
- `train/warmstart.py` (torch): `pair_to_sample`, `collate` (pads to max T, builds pad_mask, padding env_rate=1.0 so log ok), `train_step` (pure-tensor, CPU-testable), `train_warmstart(train_cfg,ym_paths)` renders via build_pair (HEAVY→2nd machine), Adam, saves checkpoint {model,in_dim,hidden} to run.extra['checkpoint'] or <cache>/warmstart_rl.pt.
- `train/__init__.py`: eager torch-free `build_targets`; torch pieces exposed via PEP562 lazy `__getattr__` (so `import audio2ay4.train` works without torch).
- CLI: `audio2ay4 train [rl|diffusion] --corpus corpus/ym --out X --batch-size --lr --max-steps --limit`. rl→train_warmstart; diffusion→stub. `_cfg` uses dataclasses.replace to set core='rl'+checkpoint.
- To load the trained core at inference: RunConfig(core='rl', extra={'checkpoint': path}); RLCore loads state.get('model',state).
- BUGFIX models/base.py: `_bootstrap()` now idempotent via module-level `_BOOTSTRAPPED` flag and ALWAYS called in get_core (was `if not _REGISTRY` — broke when 'rl' registered before 'dummy', causing KeyError 'dummy').
- Test `tests/test_train_warmstart.py` (importorskip torch): build_targets shapes, collate variable-length pad, loss finite, overfit-one-batch reduces loss<0.5*first. Synthetic random regs (NO render) → CPU-safe.
- Status: 27 tests pass (was 23), ruff clean.

## Plan A — A2 parallel render (perf)
- Corpus render in train_warmstart was single-threaded → 10-15% CPU on 32 cores, ~56min for 4961 YM. Fixed with process-pool parallel render.
- NEW `train/render.py` (torch-free): `Sample`, `pair_to_sample` (MOVED here from warmstart), `_render_one(task)` module-level worker (picklable for spawn), `resolve_workers(workers,total)` (0/None→os.cpu_count(), clamp [1,total]), `render_samples(ym_paths,run,cache_dir,*,workers=None,log_every=100)` → ProcessPoolExecutor + as_completed, preserves input order (results dict by index), skips+logs failures, prints "rendered i/total | ok N | f/s | ETA".
- warmstart.py: now `from .render import Sample, pair_to_sample, render_samples`; train_warmstart got `workers` kwarg, calls render_samples (removed old serial loop). __all__ re-exports pair_to_sample/render_samples (tests import pair_to_sample from warmstart still work).
- CLI: `train --workers N` (default 0=all cores) → train_warmstart(..., workers=args.workers).
- CRITICAL torch-free fix: workers were pulling torch because `targets.py` imports `..models.policy.spec`, and importing that submodule ran `models/policy/__init__.py` which eagerly imported torch-based core/network. FIXED: made `models/policy/__init__.py` lazy (PEP562 __getattr__ for RLCore/ReversePlayer, TYPE_CHECKING hints, no eager torch import). base.py get_core now registers 'rl' via `from .policy import core` (was `from . import policy`). Verified: `import audio2ay4.train.render` → torch NOT loaded.
- Cannot test live render here (broken CPU segfaults). Parallel logic validated by import/unit tests only; real render on 2nd machine.
- Doc updated: test_instructions/01_second-machine-warmstart.md mentions --workers.

## Plan A — A2 training-loop speedup (windowing + progress)
- Symptom on 2nd machine: render fast (12x), but training "stuck" — step 1 logged then 15min silence at ~17% CPU. Cause: each step trained on WHOLE songs (thousands of frames), collate padded to batch-max length → tens of s/step (and likely CPU not GPU).
- Fix: `_crop(sample, window, rng)` in warmstart.py — random fixed-length time window (default 512 frames); samples shorter than window returned unchanged (identity). Crops feats (T,dim)[s:e] and targets (voice (3,T)→[:,s:e], frame/env (T,)→[...,s:e]). Bounds per-step compute + uniform batches.
- train_warmstart: new `window: int|None = 512` kwarg; loop does `collate([_crop(samples[i], window, rng) for i in idx])`. Added startup banner `Training on {device}: {N} tunes | batch {bs} | window {w} | {steps} steps` (so user can SEE cpu vs cuda). Progress now logs steps 1-3 + every log_every + last, with `it/s` and `ETA` (time.monotonic t0).
- CLI: `train --window N` (default 512, 0→whole songs). Passed as `window=(args.window or None)`.
- If banner shows cpu → torch can't see GPU (CUDA wheel mismatch); reinstall cu121 torch.
- Test: tests/test_train_warmstart.py::test_crop_windows_long_and_keeps_short (crop shapes + short-sample identity). 28 tests pass, ruff clean.

## Plan A — A2 first real run analysis + convergence tooling
- 2nd machine ran full pipeline OK: render 4942/4961 ok (19 skipped: MIX1/YMT1/YMT2 magic, LHA Huffman errors, buffer-too-small, list-index — all foreign/corrupt formats, expected), 24.5 f/s on 32 workers (~3min). Training 12.4 it/s on cuda (GTX 1070, torch 2.5.1+cu121). Checkpoint 3.1MB.
- DIAGNOSIS of poor result: loss plateaued ~500 by step ~50, never improved. pitch MSE ~250 (=RMS ~16 semitones), volume MSE ~250 (~16 dB) ≈ corpus per-head VARIANCE → model collapsed to predicting per-head MEANS, not learning audio→register map. Root causes: (1) only ~0.67 epoch (2000 steps×16 ≈ 32k windows vs ~48k for 1 epoch), (2) lr=1e-4 too low (overfit test fits at 1e-2), (3) raw per-step batch loss is too noisy to read.
- FIX (observability + optimization) in warmstart.py:
  * `_lr_at(step, base_lr, max_steps, warmup)`: linear warmup (max(1,min(500,max_steps//10))) then cosine decay to ~0. Set per-step on opt.param_groups.
  * EMA of train loss (`avg=`, 0.98 decay) in log line + `lr=`.
  * Held-out validation: last `val_frac` (default 0.05) tunes → `val_samples`; `_evaluate(net, val_samples, ...)` @torch.no_grad eval-mode mean loss over n_batches=8, logged every `val_every` (250) + final as `|| val=`. Loop trains on `train_samples` only.
  * Banner now shows `N train / M val tunes | ... | lr X (warmup W, cosine)`.
  * train_warmstart new kwargs: `val_frac=0.05, val_every=250`.
- CLI: `--lr` default raised 1e-4→3e-4; added `--val-frac` (0.05); passed `val_frac=args.val_frac`.
- Recommendation given to user: real run needs `--max-steps 50000 --batch-size 64 --lr 3e-4` (~70min on GTX1070); watch avg/val fall, want pitch/volume MSE well below ~50.
- Test: test_lr_schedule_warmup_then_cosine_to_zero. 29 tests pass, ruff clean.
- NOTE: default --max-steps still 2000 (smoke); real warm-start must override. env_rate loss noisy (envelope rare in corpus); occasionally 0.000 when batch has no env-active frames — expected.

## Plan A — A2 cache-hit reporting
- User asked "is it using cache?" — render output didn't say. Cache mechanism (data/pairing.py): build_pair → `.cache/{file_sha1}_{cfg_key}.npz`, cfg_key=sha1(v{_CACHE_VERSION}:sample_rate:frame_rate_hz:feat_kind)[:12]. Only sample_rate/frame_rate_hz/feat_kind affect key — NOT batch/lr/max-steps/window, so re-runs with diff hyperparams DO hit cache. First completed run populates ~4942 npz; subsequent runs should be cache hits.
- Added `is_cached(ym_path, cfg, cache_dir)->bool` in pairing.py (existence check on computed cache path; corrupt entry counts as hit but build_pair re-renders).
- render.py: `_render_one` now returns `(Sample, was_cached)`; render_samples tracks cached_n. Banner: `Rendering N YM files (workers=W, cache=.cache) …`. Progress: `ok N (cache C / new R)`. Final: `Render done: N ok (C from cache, R freshly rendered), S skipped.` Both serial+parallel paths.
- Likely truth: 2nd machine WAS hitting cache; ~24 f/s ceiling is IPC return of ~1.5MB Samples (deserialized serially in main) + build_targets/parse_song per-frame Python, NOT emulator render. Indicator now proves it. If still slow with cache hits, next optim = reduce IPC (workers return only cache-populated signal, main loads lazily) — NOT done yet.
- 29 tests pass, ruff clean, render import still torch-free.

## Plan A — A2 pitch head: regression → classification
- Reason: warm-start val plateaued ~450 while train fell (overfit); pitch MSE ~180 = ~13 semitone RMS (mean-collapse). Switched pitch from tanh-bounded MSE regression to softmax over 1-semitone bins.
- spec.py: added PITCH_MIN/PITCH_MAX (30/90), N_PITCH_BINS=61, PITCH_BIN_WIDTH, pitch_to_bin(semitones)->int, bin_to_pitch(b)->float (torch-free).
- network.py: head_pitch now Conv1d(hidden, N_VOICES*N_PITCH_BINS); forward returns `pitch_logits` (B,V,K,T) via view. (was `pitch` (B,V,T)).
- targets.py: emits `pitch_bin` (3,T) int64 via pitch_to_bin (was continuous `pitch`); default fill = pitch_to_bin(PITCH_CENTER).
- warmstart_loss.py: pitch term = masked cross_entropy over bins; logits permute(0,2,1,3)->(B,K,V,T) vs target (B,V,T); masked by tone*pad. parts key still "pitch" (now CE nats, random≈ln(61)≈4.1).
- warmstart.py collate: `pitch_bin` int64 path, fill (N_PITCH_BINS-1)//2; assert max<N_PITCH_BINS.
- core.py _decode: pitch = PITCH_MIN + argmax(pitch_logits,axis=1)*PITCH_BIN_WIDTH.
- tests updated (test_train_warmstart: pitch_bin shapes/dtype/range; test_policy_rl: pitch_logits (2,3,N_PITCH_BINS,17)). doc loss note updated (pitch=CE).
- IMPORTANT: old continuous checkpoints are INCOMPATIBLE (head_pitch 3→183 outputs) — must retrain fresh. Render CACHE still valid (stores feats+regs; targets rebuilt in pair_to_sample), so NO re-render needed.
- 29 tests pass, ruff clean. Next: if pitch CE falls but volume MSE stays stuck (~250=16dB), apply same bin-classification to volume.

## Plan A — A2 permutation-invariant voice loss (uPIT)
- 3rd run (classification head) showed pitch CE STUCK ~2.8 even on TRAIN (random 4.1, good<1) = only learned marginal pitch histogram; volume MSE drove train down but val flat ~190 (overfit volume). Diagnosis: per-voice pitch unlearnable because 3 AY tone channels are identical-timbre → corpus's arbitrary A/B/C labelling not recoverable from mono mix.
- Fix: utterance-level PIT in warmstart_loss.py. Added `_PERMS=itertools.permutations(range(N_VOICES))` (6) + `_best_voice_perm(heads,targets,pad_mask,w)->(B,V) target idx`: builds (B,Vp,Vt) cost m (pitch CE via log_softmax+gather, volume SE, tone/noise/env_use pairwise BCE, masked by target-j gates*pad, weighted), enumerates 6 perms, argmin/batch. warmstart_loss wraps selection in no_grad, gathers permuted per-voice targets (pitch_bin/volume/tone/noise/env_use) via gather(1, perm.unsqueeze(-1).expand), runs SAME per-head reductions with permuted targets. Global heads untouched. Signature unchanged.
- Per-SEQUENCE perm (one/tune) not per-frame → voice identity consistent over time.
- Test: test_warmstart_loss_invariant_to_voice_relabelling (roll target voices → loss approx unchanged). 30 tests pass, ruff clean.
- IMPORTANT: loss-only → render CACHE still valid, NO re-render. Checkpoint still must be fresh (earlier head change).
- If pitch still floors high after PIT → next suspect = features (mel lacks semitone resolution): switch feat_kind to CQT (changes cfg_key → full re-render).
- v3 RESULT (8k steps, cuda, checkpoints/warmstart_rl_v3.pt): PIT WORKS. pitch CE 5.53→2.64 (broke 2.8 floor, still descending). val 153→111 (best @6500). train avg→75. Per-head @8k: volume MSE 63 (~85% of weighted total!), env_rate 11 (0.5w=5.5), pitch 2.64, tone/noise/env_use<0.25.
- CAUTION: val gain rate collapsed after step 5000 (115→111 over 1500 steps) while train avg fell 90→75 = ONSET OF OVERFIT, started while LR still 1e-4 (not just schedule). Gap train75/val113. The ceiling is now VOLUME generalization, not optimization. More steps (50k) will overfit volume further, val maybe ~105 floor.
- NEXT LEVER (highest, cheap, loss-only/no-rerender): VOLUME → classification, mirror pitch fix. DAC is 4-bit/16 levels → regressing dB is wrong objective. Softmax over 16 DAC levels should close volume generalization gap like pitch did. THEN CQT for pitch<2 (re-render). Optional: small weight_decay/dropout to slow volume overfit.

## Decision log (latest)
- Q: run full 50K warm-start now? A: NO. The 8k run already exposed the ceiling (volume generalization). 50k of the same recipe mostly overfits volume; val would floor ~105. Do the volume→classification fix FIRST (loss-only, reuses cache), THEN run long.
- NEXT ACTION when picked up: implement volume → 16-level DAC classification (mirror the pitch change): spec.py N_VOL_LEVELS=16 + db<->dac helpers; network.py head_volume → Conv1d(hidden, N_VOICES*16) reshape (B,3,16,T) `volume_logits`; targets.py+warmstart.py emit `volume_level` int64; warmstart_loss.py volume term → masked CE + update PIT cost matrix to score volume by CE; core.py _decode argmax→dB; update tests. Old checkpoints already incompatible → retrain fresh, cache still valid (no re-render).

## NOT yet done (next options)
- A3: chip/diff differentiable emulator (DDSP square+noise+env, validate vs trusted emulator). A4: Regime-1 reward training (analysis-by-synthesis, eval.spectral_distance + embedding + chroma/onset). A5: real/augmented (Suno) reward phase + jitter/idiomatic regularizers. A6: optional PPO.
- Run actual A2 warm-start on 2nd machine: `audio2ay4 train rl --corpus corpus/ym` (renders 4961 tunes → cache → trains). Then eval with core=rl+checkpoint.
- Plan B diffusion. Augmentation in data/. CQT/EnCodec adapters. samples/ CI gates.
