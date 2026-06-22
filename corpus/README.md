# corpus/

Place your YM training data here. The data pipeline (`audio2ay4.data.scan_corpus`) scans this
folder **recursively** for `*.ym` files, deduplicates by content, and renders each through the
audio2ay3 emulator to build `(audio, registers)` training pairs.

```
corpus/
└─ ym/        # drop .ym files (any nested structure) here
```

Bulk data (`*.ym`, archives) is git-ignored; only this README and the folder structure are tracked.

## Sources (verified 2026-06-21)

Prefer the **YM** register-stream format (read natively by audio2ay3: YM2/3/3b/5/6 + LHA depack).
Other AY formats (`.ay`, `.vtx`, `.psg`, `.pt3`, `.vgm`) must be converted to `.ym` first.

| Source | Format | Notes |
|--------|--------|-------|
| **Modland — YM collection** — https://modland.com/pub/modules/YM/ | `.ym` native | **Primary.** Thousands of tunes, organised by author. Directly downloadable. |
| **Project AY (World of Spectrum)** — https://worldofspectrum.org/projectay/ | `.ay` | Huge ZX/Amstrad AY archive; convert `.ay → .ym`. |
| **ZX-Art** — https://zxart.ee/eng/music/ | `.pt3` `.psg` `.ay` `.stc` | Large, actively updated AY database; convert to `.ym`. |
| **VGMRips — AY-3-8910 packs** — https://vgmrips.net/packs/chip/ay-3-8910 | `.vgm` | Clean per-chip register logs; convert `.vgm → .ym`. |
| **ST-Sound / Leonard (Oxygene)** — http://leonard.oxg.free.fr/ | `.ym` | Origin of the YM format; Atari ST YM music + format spec. |

### Bulk-fetch Modland YM (native `.ym`, recommended start)

Use the bundled downloader [`scripts/download_modland_ym.py`](../scripts/download_modland_ym.py)
(stdlib-only, no extra dependencies). It reads the path manifest
[`scripts/modland_ym.txt`](../scripts/modland_ym.txt) — 4961 entries of the form
`YM/<author>/<tune>.ym` — and fetches each from `https://modland.com/pub/modules/` into
`corpus/ym/<author>/<tune>.ym` (the leading `YM/` is stripped).

```powershell
# full corpus (~17 MB, 4961 files); re-run anytime — existing files are skipped (resume)
.\.venv\Scripts\python.exe scripts\download_modland_ym.py --workers 8

# smoke-test a handful first
.\.venv\Scripts\python.exe scripts\download_modland_ym.py --limit 5
```

Features: concurrent workers, resume (skips already-downloaded files), atomic `.part` writes,
retry-with-backoff, and a `corpus/ym/_failures.log` listing any files that failed (re-run to retry).
Flags: `--list` (manifest path), `--out` (default `corpus/ym`), `--workers`, `--timeout`,
`--retries`, `--limit`.

> Alternative (needs `wget`): `wget -r -np -nH --cut-dirs=3 -A ".ym" https://modland.com/pub/modules/YM/ -P corpus/ym`


## Notes

- Filter to AY-3-8910 / YM2149 tunes (the corpus may contain other chips).
- These are third-party tunes — respect each archive's licensing/usage terms.
