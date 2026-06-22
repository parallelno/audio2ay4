"""Download Modland YM tunes listed in a path file into the corpus folder.

The list file (e.g. ``modland_ym.txt``) holds one relative path per line, like
``YM/4-Mat/chuck rock.ym``. Each becomes ``https://modland.com/pub/modules/<url-encoded path>``.
Files are saved under ``--out`` mirroring the author structure (the leading ``YM/`` is stripped),
so ``YM/4-Mat/chuck rock.ym`` → ``corpus/ym/4-Mat/chuck rock.ym``.

Stdlib only. Safe to re-run: existing non-empty files are skipped (resume), downloads are written
atomically via a ``.part`` temp file, and failures are retried then logged.

Usage:
    python scripts/download_modland_ym.py                      # full download into corpus/ym
    python scripts/download_modland_ym.py --limit 5            # smoke test
    python scripts/download_modland_ym.py --workers 8          # tune concurrency
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_BASE = "https://modland.com/pub/modules/"
USER_AGENT = "audio2ay4-corpus-fetch/1.0 (+local research; polite)"


def _read_paths(list_path: str) -> list[str]:
    with open(list_path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _local_path(out_dir: str, rel: str) -> str:
    # Strip a leading "YM/" so files land directly under out_dir/<author>/...
    parts = rel.split("/")
    if parts and parts[0].upper() == "YM":
        parts = parts[1:]
    return os.path.join(out_dir, *parts)


def _url_for(base: str, rel: str) -> str:
    return base + urllib.parse.quote(rel, safe="/")


def _download_one(base: str, out_dir: str, rel: str, timeout: int, retries: int) -> tuple[str, str]:
    """Return (status, rel) where status is 'ok' | 'skip' | 'fail: <reason>'."""
    dst = _local_path(out_dir, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return ("skip", rel)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = _url_for(base, rel)
    tmp = dst + ".part"
    last_err = "unknown"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                last_err = "empty response"
                raise ValueError(last_err)
            with open(tmp, "wb") as out:
                out.write(data)
            os.replace(tmp, dst)
            return ("ok", rel)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError, OSError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 10.0))
    return (f"fail: {last_err}", rel)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download Modland YM files from a path list.")
    default_list = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modland_ym.txt")
    p.add_argument("--list", default=default_list, help="path-list file (one rel path per line)")
    p.add_argument("--out", default=os.path.join("corpus", "ym"), help="output directory")
    p.add_argument("--base", default=DEFAULT_BASE, help="base URL the paths are relative to")
    p.add_argument("--workers", type=int, default=6, help="concurrent downloads (be polite)")
    p.add_argument("--timeout", type=int, default=30, help="per-request timeout (s)")
    p.add_argument("--retries", type=int, default=3, help="attempts per file")
    p.add_argument("--limit", type=int, default=0, help="only the first N entries (0 = all)")
    args = p.parse_args(argv)

    if not os.path.exists(args.list):
        print(f"List file not found: {args.list}", file=sys.stderr)
        return 2

    paths = _read_paths(args.list)
    if args.limit > 0:
        paths = paths[: args.limit]
    total = len(paths)
    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {total} files -> {args.out} ({args.workers} workers)")

    ok = skip = fail = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_download_one, args.base, args.out, rel, args.timeout, args.retries): rel
            for rel in paths
        }
        for i, fut in enumerate(as_completed(futs), 1):
            status, rel = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failures.append((rel, status))
            if i % 100 == 0 or i == total:
                rate = i / max(1e-6, time.time() - t0)
                print(f"  {i}/{total}  ok={ok} skip={skip} fail={fail}  ({rate:.1f}/s)")

    if failures:
        log = os.path.join(args.out, "_failures.log")
        with open(log, "w", encoding="utf-8") as fh:
            for rel, status in failures:
                fh.write(f"{status}\t{rel}\n")
        print(f"{fail} failures logged to {log} (re-run to retry them).")

    print(f"Done in {time.time() - t0:.0f}s: ok={ok} skip={skip} fail={fail} of {total}.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
