# Task: run the Plan A warm-start training for audio2ay4 (second / training machine)

> Temporary working note (not final documentation). Setup guidance for the GPU/training
> machine that runs the heavy emulator-render + warm-start. The dev machine has a broken CPU
> and cannot render YM; this machine is the one that does the real work.

You are setting up audio2ay4 on a fresh GPU machine and running the supervised
warm-start (Plan A phase A2). Unlike the dev machine, THIS machine has a healthy CPU,
so emulator rendering of the YM corpus is expected to work normally.

Reference docs already in the repo: README is minimal; dependency extras are in
`pyproject.toml`, corpus details in `corpus/README.md`, design in `design/README.md`.

## 0. Prerequisites
- Python 3.12+  (`python --version`)
- git  (the `ay3` extra installs audio2ay3 from GitHub)
- An NVIDIA GPU + driver if you want CUDA (optional; CPU works, just slower)

## 1. Get the code (corpus is committed in-repo)
```bash
git clone https://github.com/parallelno/audio2ay4
cd audio2ay4
# sanity: should print ~4961
#  PowerShell: (Get-ChildItem -Recurse corpus/ym -Filter *.ym).Count
#  bash:       find corpus/ym -name '*.ym' | wc -l
```

## 2. Create and activate a virtual environment (Python 3.12)
```powershell
# Windows PowerShell:
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
```
```bash
# Linux/macOS:
python3.12 -m venv .venv && source .venv/bin/activate
```
```bash
python -m pip install -U pip
```

## 3. Install PyTorch FIRST (so the right build is used)
```bash
# GPU (pick the index URL matching your CUDA, cu121 is a safe common default):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU-only fallback:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 4. Install audio2ay4 with all needed extras
```bash
# neural = torch stack, ay3 = the audio2ay3 emulator (from git), dev = pytest/ruff
pip install -e ".[neural,ay3,dev]"
```

## 5. Verify the install
```bash
pytest -q                                   # expect 27 passed
python -c "import torch; print('cuda:', torch.cuda.is_available())"
# Confirm native rendering works on this machine (no model involved):
audio2ay4 validate samples/ym/song01.ym -o song01.wav
```

## 6. Smoke-test the training end-to-end (renders only a few tunes)
```bash
audio2ay4 train rl --corpus corpus/ym --limit 8 --max-steps 50
# Expect: per-step loss lines, then "Saved warm-start checkpoint -> .cache/warmstart_rl.pt"
```

## 7. Full warm-start run
```bash
audio2ay4 train rl --corpus corpus/ym --out checkpoints/warmstart_rl.pt
# Defaults: batch-size 16, lr 1e-4, max-steps 2000. Override as needed, e.g.
# audio2ay4 train rl --corpus corpus/ym --max-steps 20000 --batch-size 32 \
#   --out checkpoints/warmstart_rl.pt
# GPU is selected automatically when available.
```

## 8. Report back
- Whether step 5 (pytest / cuda / validate render) all succeeded.
- The smoke-test loss values and the final-run loss trajectory (start vs end, and the
  per-head parts: pitch/volume/tone/noise/env_use/noise_pitch/env_rate/env_shape/env_retrig).
- The saved checkpoint path and size.
- Any tune that failed to render (these are logged), and your CUDA/torch versions.

## Notes / gotchas
- First run renders the whole corpus through the emulator and caches pairs under
  `.cache/` (npz). Subsequent runs reuse the cache, so only step 7's first invocation
  is slow.
- To later use the trained core for inference: it loads from a RunConfig with
  `core="rl"` and `extra={"checkpoint": "<path>"}`.
- If `pip install -e ".[ay3]"` fails to build audio2ay3, install a C/C++ build
  toolchain (Linux: build-essential; Windows: VS Build Tools) and retry.
