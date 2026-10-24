#!/usr/bin/env python3
"""L5a baseline: perplexity of GGUF models on a fixed eval sample.

Runs llama-perplexity (llama.cpp) against tools/eval/wikitext_sample.txt and
prints PPL, delta-PPL vs the first model, tokens, wall time, tokens/s and peak
RSS. Numbers are only comparable between runs that share the same eval file,
--ctx and llama-perplexity build; the script prints all three so a result can
be reproduced later.

Usage:
    .venv/bin/python tools/ppl_baseline.py \
        --gguf ~/.vipera/models/NousResearch_Hermes-4-14B-Q4_K_M.gguf \
        --gguf ~/.vipera/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

Run it with the vipera venv so psutil is available for peak-RSS sampling;
without psutil the script still works, it just cannot report peak RSS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = ROOT / "tools" / "eval" / "wikitext_sample.txt"
DEFAULT_BIN_DIRS = (
    Path.home() / ".vipera" / "bin",
    Path.home() / "dev" / "llama.cpp" / "build" / "bin",
    Path.home() / "llama.cpp" / "build" / "bin",
    Path("/usr/local/bin"),
)

# llama-perplexity writes to stderr via LOG_INF; we merge it into stdout.
RE_PPL = re.compile(r"Final estimate: PPL = ([0-9.]+)(?: \+/- ([0-9.]+))?")
# The -f (file) path logs this shape; the BPE path differs and is not supported.
RE_CHUNKS = re.compile(
    r"calculating perplexity over (\d+) chunks, n_ctx=(\d+), batch_size=(\d+), n_seq=(\d+)"
)
RE_PASS_S = re.compile(r"([\d.]+) seconds per pass")

try:
    import psutil
except ImportError:
    psutil = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_binary(bin_dir: str | None) -> Path:
    dirs = [Path(bin_dir).expanduser()] if bin_dir else list(DEFAULT_BIN_DIRS)
    for d in dirs:
        candidate = d / "llama-perplexity"
        if candidate.is_file():
            return candidate
    sys.exit(
        "llama-perplexity not found (searched: "
        + ", ".join(str(d) for d in dirs)
        + "); build it or pass --bin-dir"
    )


def run_one(binary: Path, gguf: Path, eval_file: Path, ctx: int, chunks: int, ngl: int) -> dict:
    # --chunks caps the PPL windows (llama default -1 = every window, which on
    # a 14B model over even a small sample can run for hours on this box).
    cmd = [
        str(binary),
        "-m",
        str(gguf),
        "-f",
        str(eval_file),
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "-ngl",
        str(ngl),
    ]
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak_rss = 0
    lines: list[str] = []
    sampler = psutil.Process(proc.pid) if psutil else None
    if sampler:
        try:
            peak_rss = sampler.memory_info().rss
        except psutil.Error:
            sampler = None
    try:
        for line in proc.stdout:
            lines.append(line)
            if sampler:
                try:
                    peak_rss = max(peak_rss, sampler.memory_info().rss)
                except psutil.Error:
                    sampler = None  # exited mid-read
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    wall = time.monotonic() - started

    output = "".join(lines)
    m_ppl = RE_PPL.search(output)
    m_chunk = RE_CHUNKS.search(output)
    m_pass = RE_PASS_S.search(output)
    result = {
        "model": gguf.name,
        "path": str(gguf),
        "sha256": sha256(gguf),
        "size_gib": round(gguf.stat().st_size / 1024**3, 3),
        "rc": proc.returncode,
        "wall_s": round(wall, 1),
        "ctx": ctx,
        "ngl": ngl,
        "chunks": int(m_chunk.group(1)) if m_chunk else None,
        "n_seq": int(m_chunk.group(4)) if m_chunk else None,
        # The -f path never prints the token count; derive scored tokens from
        # the chunking (each window scores ctx-1 tokens).
        "scored_tok": (int(m_chunk.group(1)) * (int(m_chunk.group(2)) - 1) if m_chunk else None),
        "pass_s": float(m_pass.group(1)) if m_pass else None,
        "ppl": float(m_ppl.group(1)) if m_ppl else None,
        "ppl_stderr": float(m_ppl.group(2)) if m_ppl and m_ppl.group(2) else None,
        "peak_rss_gib": round(peak_rss / 1024**3, 3) if peak_rss else None,
    }
    # "ppl tok/s": tokens covered by ONE perplexity pass (n_seq * ctx) per the
    # first-pass timing llama.cpp prints. A prefill-style rate — NOT comparable
    # with the generation T/s that `vipera bench run` reports.
    if result["pass_s"] and result["n_seq"]:
        result["ppl_tok_per_s"] = round(result["n_seq"] * ctx / result["pass_s"], 2)
    else:
        result["ppl_tok_per_s"] = None
    if result["ppl"] is None:
        result["log_tail"] = output[-800:]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", action="append", required=True, help="Model to measure (repeatable)")
    ap.add_argument("--file", default=str(DEFAULT_EVAL), help="Eval text (keep byte-identical)")
    ap.add_argument("--bin-dir", default=None, help="Directory with llama-perplexity")
    ap.add_argument("--ctx", type=int, default=512, help="Context for the PPL window")
    ap.add_argument(
        "--chunks",
        type=int,
        default=20,
        help="Max PPL windows (llama default -1 processes all; keep fixed to compare runs)",
    )
    ap.add_argument(
        "--ngl", type=int, default=0, help="GPU layers; 0 = pure CPU (fixed for stable PPL)"
    )
    ap.add_argument("--json", default=None, help="Also write results as JSON here")
    args = ap.parse_args()

    binary = find_binary(args.bin_dir)
    eval_file = Path(args.file).expanduser()
    if not eval_file.is_file():
        sys.exit(f"eval file not found: {eval_file}")

    v = subprocess.run([str(binary), "--version"], capture_output=True, text=True)
    version = (v.stdout + v.stderr).splitlines()
    print(f"binary : {binary}")
    print(f"version: {version[0] if version else 'unknown'}")
    print(
        f"eval   : {eval_file} (sha256 {sha256(eval_file)[:12]}, {eval_file.stat().st_size} bytes)"
    )
    print(f"ctx    : {args.ctx}  chunks: {args.chunks}  ngl: {args.ngl}\n")
    if not psutil:
        print("note: psutil missing -> peak RSS will not be reported\n")

    results = []
    for i, path in enumerate(args.gguf):
        gguf = Path(path).expanduser()
        if not gguf.is_file():
            sys.exit(f"model not found: {gguf}")
        print(f"running {gguf.name} ...", flush=True)
        r = run_one(binary, gguf, eval_file, args.ctx, args.chunks, args.ngl)
        r["ppl_delta"] = (
            round(r["ppl"] - results[0]["ppl"], 4)
            if r["ppl"] is not None and results and results[0]["ppl"] is not None
            else (0.0 if i == 0 else None)
        )
        results.append(r)

    print(
        f"\n{'model':<44}{'PPL±stderr':>16}{'ΔPPL':>9}"
        f"{'tok':>8}{'ppl tok/s':>10}{'peakGiB':>9}{'sha256':>14}"
    )
    for r in results:
        if r["ppl"] is None:
            print(
                f"{r['model']:<44}{'FAILED':>16} rc={r['rc']} tail: {r.get('log_tail', '')[-200:]}"
            )
            continue
        err = r["ppl_stderr"] or 0.0
        # Pre-3.12 f-strings cannot reuse the outer quote, and this repo
        # supports Python 3.11.
        ppl_cell = f"{r['ppl']:.4f}±{err:.4f}"
        print(
            f"{r['model'][:43]:<44}{ppl_cell:>16}"
            f"{r['ppl_delta']:>+9.4f}{(r['scored_tok'] or 0):>8}{(r['ppl_tok_per_s'] or 0.0):>10.1f}"
            f"{(r['peak_rss_gib'] or 0):>9.3f}{r['sha256'][:12]:>14}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\njson: {args.json}")
    return 0 if all(r["ppl"] is not None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
