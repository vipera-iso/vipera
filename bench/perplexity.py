"""Run llama-perplexity and record a quality number next to the speed numbers (S10).

`bench/bench.py` measures how *fast* a model runs; this measures how *good* it is.
vipera's scheduler does not model quality at all, so this is the other axis: it
establishes a baseline, and it is the harness any compression/pruning experiment
has to be judged with.

    python3 -m bench.perplexity \\
        --bin ~/dev/llama.cpp/build/bin/llama-perplexity \\
        --model models/olmoe-q4km.gguf \\
        --file ~/dev/llama.cpp/wikitext-2-raw/wiki.test.raw \\
        --chunks 20 --json bench/results/olmoe-ppl.json

Anchoring the corpus matters: perplexity is only comparable across runs that use
the same text, context and chunk count. The canonical choice is wikitext-2 test
(`llama.cpp/scripts/get-wikitext-2.sh`); use the whole set for a publishable
number, and a fixed `--chunks` for a quick regression.

Exit codes: 0 measured, 1 the run failed or no PPL was printed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

DEFAULT_BIN = "llama-perplexity"

# "Final estimate: PPL = 8.6426 +/- 0.30373"
PPL_RE = re.compile(r"PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def parse_ppl(text: str) -> tuple[float, float] | None:
    matches = PPL_RE.findall(text)
    if not matches:
        return None
    value, stderr = matches[-1]
    return float(value), float(stderr)


def run_perplexity(binary: str, model: str, *, corpus: str, chunks: int | None = None,
                   context: int = 512, threads: int | None = None, ngl: int | None = 0,
                   timeout: float | None = None) -> dict:
    cmd = [binary, "-m", model, "-f", corpus, "-c", str(context)]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]

    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "command": cmd,
                "elapsed_s": round(time.monotonic() - started, 3)}
    elapsed = time.monotonic() - started

    report = {"ok": False, "exit_code": proc.returncode, "command": cmd,
              "elapsed_s": round(elapsed, 3), "chunks": chunks,
              "context": context, "corpus": os.path.basename(corpus)}
    ppl = parse_ppl(proc.stdout + proc.stderr)
    if ppl is None:
        report["error"] = "no PPL line in output"
        report["stderr_tail"] = proc.stderr[-2000:]
        return report
    report["ok"] = proc.returncode == 0
    report["ppl"], report["ppl_stderr"] = ppl
    return report


def format_table(report: dict) -> str:
    lines = [f"ok            {report.get('ok')}",
             f"exit          {report.get('exit_code')}",
             f"elapsed       {report.get('elapsed_s')} s"]
    if report.get("ppl") is not None:
        lines.append(f"corpus        {report['corpus']}  ({report['chunks']} chunks, "
                     f"ctx {report['context']})")
        lines.append(f"PPL           {report['ppl']:.4f} +/- {report['ppl_stderr']:.5f}")
    if report.get("error"):
        lines.append(f"error         {report['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="perplexity", description=__doc__)
    p.add_argument("--bin", default=DEFAULT_BIN)
    p.add_argument("--model", required=True)
    p.add_argument("--file", required=True, dest="corpus", help="text corpus")
    p.add_argument("--chunks", type=int, default=None,
                   help="limit chunks (omit for the full set)")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--ngl", type=int, default=0)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--json", dest="json_path")
    args = p.parse_args(argv)

    binary = shutil.which(args.bin) or args.bin
    report = run_perplexity(binary, args.model, corpus=args.corpus, chunks=args.chunks,
                            context=args.context, threads=args.threads, ngl=args.ngl,
                            timeout=args.timeout)
    print(format_table(report))
    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"report written to {args.json_path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
