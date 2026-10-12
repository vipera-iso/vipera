"""Capture a real MoE routing trace from a llama.cpp build (S7).

The scheduler is only as honest as its routing input: everything downstream —
hit rate, bytes/token, tok/s — rests on a modelled Zipf skew until a real trace
exists. This runs `llama-cli` with `LLAMA_MOE_TRACE` set (a local fork patch that
writes the router's selected experts per decode step) and leaves the JSONL in the
format `vipera.trace.load_trace` reads:

    {"t": 0, "layers": [[e0, e1, ...], [...], ...]}

    python3 -m bench.capture_trace \\
        --bin ~/dev/llama.cpp/build/bin/llama-cli \\
        --model /path/to/olmoe-q4km.gguf \\
        --tokens 256 --out bench/results/olmoe.jsonl

The llama.cpp build must be patched for capture; this reports a clear error if
`LLAMA_MOE_TRACE` produced no file, rather than silently writing an empty trace.

Exit codes: 0 captured, 1 the run or capture failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

DEFAULT_BIN = "llama-cli"
DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8


def build_command(binary: str, model: str, *, prompt: str, tokens: int,
                  threads: int | None, ngl: int | None, seed: int,
                  extra: list[str]) -> list[str]:
    # `-st` (single-turn) makes llama-cli generate from the prompt and exit, so
    # the run is scriptable; there is no conversation loop to feed.
    cmd = [binary, "-m", model, "-p", prompt, "-n", str(tokens),
           "-st", "--temp", "0", "--seed", str(seed)]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    cmd += extra
    return cmd


def summarise(path: str) -> dict:
    """Shape and skew of a captured trace, without importing the scheduler."""
    tokens = 0
    layers = None
    top_k = None
    per_layer: dict[int, set[int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tok_layers = rec["layers"]
            if layers is None:
                layers = len(tok_layers)
            if top_k is None:
                top_k = len(tok_layers[0])
            for li, experts in enumerate(tok_layers):
                per_layer.setdefault(li, set()).update(experts)
            tokens += 1
    return {
        "tokens": tokens,
        "layers": layers,
        "top_k": top_k,
        "distinct_experts_per_layer": [len(per_layer.get(i, set())) for i in range(layers or 0)],
    }


def capture(binary: str, model: str, *, prompt: str = DEFAULT_PROMPT, tokens: int = 256,
            threads: int | None = None, ngl: int | None = None, seed: int = 0,
            extra: list[str] | None = None, timeout: float | None = None,
            out: str | None = None) -> dict:
    cmd = build_command(binary, model, prompt=prompt, tokens=tokens, threads=threads,
                        ngl=ngl, seed=seed, extra=extra or [])

    with tempfile.TemporaryDirectory() as tmp:
        trace = os.path.join(tmp, "routing.jsonl")
        env = dict(os.environ, LLAMA_MOE_TRACE=trace)

        started = time.monotonic()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "elapsed_s": time.monotonic() - started}
        elapsed = time.monotonic() - started

        if not os.path.exists(trace) or os.path.getsize(trace) == 0:
            return {
                "ok": False,
                "error": "no routing trace was written — is the llama.cpp build patched "
                         "for LLAMA_MOE_TRACE?",
                "exit_code": proc.returncode,
                "elapsed_s": round(elapsed, 3),
                "stderr_tail": proc.stderr[-2000:],
            }

        info = summarise(trace)
        info.update({"ok": True, "exit_code": proc.returncode,
                     "elapsed_s": round(elapsed, 3), "command": cmd,
                     "bytes": os.path.getsize(trace)})
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            shutil.copyfile(trace, out)
            info["written"] = out
    return info


def format_table(info: dict) -> str:
    lines = [f"ok            {info.get('ok')}"]
    if "exit_code" in info:
        lines.append(f"exit          {info['exit_code']}")
    lines.append(f"elapsed       {info.get('elapsed_s')} s")
    if info.get("ok"):
        lines.append(f"tokens        {info['tokens']}")
        lines.append(f"layers        {info['layers']}")
        lines.append(f"top_k         {info['top_k']}")
        seen = info["distinct_experts_per_layer"]
        lines.append(f"distinct/layer min {min(seen)} max {max(seen)}")
    if info.get("written"):
        lines.append(f"written       {info['written']}")
    if info.get("error"):
        lines.append(f"error         {info['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="capture_trace", description=__doc__)
    p.add_argument("--bin", default=DEFAULT_BIN, help="llama-cli executable")
    p.add_argument("--model", required=True, help="MoE model in GGUF format")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--tokens", type=int, default=256, help="decode steps to capture")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--ngl", type=int, default=None, help="layers offloaded to GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--out", help="write the trace here")
    p.add_argument("extra", nargs="*", help="extra flags passed through to llama-cli")
    args = p.parse_args(argv)

    binary = shutil.which(args.bin) or args.bin
    info = capture(binary, args.model, prompt=args.prompt, tokens=args.tokens,
                   threads=args.threads, ngl=args.ngl, seed=args.seed,
                   extra=args.extra, timeout=args.timeout, out=args.out)
    print(format_table(info))
    return 0 if info.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
