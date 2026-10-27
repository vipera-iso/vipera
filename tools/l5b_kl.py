#!/usr/bin/env python3
"""Measure the KL divergence of a quantized GGUF against its F16/BF16 reference (L5b).

Reference must be the highest-precision file for the same model (BF16 or F16,
never another quant). Deliberately not wired into `vipera bench`: on a 14B
CPU-only this run is 5-15 minutes.

    python tools/l5b_kl.py --base base-f16.gguf --target q4km.gguf \\
        --eval eval.txt --label q4km-vs-f16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from vipera.optimizer import OptimizerError, find_llama_tool

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")
_KL_RE = re.compile(r"Mean\s+KLD:\s*([\d.eE+-]+)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _supports_kl(binary: str) -> bool:
    try:
        proc = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--kl-divergence" in f"{proc.stdout}\n{proc.stderr}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="reference F16/BF16 GGUF")
    parser.add_argument("--target", required=True, type=Path, help="quantized GGUF")
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--n-seq", type=int, default=4, dest="n_seq")
    parser.add_argument("--out-dir", type=Path, default=Path("tools/l5b_results"))
    parser.add_argument("--bin")
    args = parser.parse_args()

    binary = args.bin
    if not binary:
        try:
            binary = find_llama_tool("perplexity")
        except OptimizerError as exc:
            print(f"{exc}")
            return 1

    if not _supports_kl(binary):
        print(
            f"{binary} does not advertise --kl-divergence; rebuild llama.cpp "
            "from a recent checkout with tools enabled."
        )
        return 1

    command = [
        binary,
        "-m",
        str(args.target),
        "-md",
        str(args.base),
        "-f",
        str(args.eval),
        "-c",
        str(args.ctx),
        "--chunks",
        str(args.chunks),
        "-ngl",
        str(args.ngl),
        "--kl-divergence",
    ]
    if args.n_seq > 1:
        command.extend(["-n-seq", str(args.n_seq)])

    t0 = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    output = f"{proc.stdout}\n{proc.stderr}"

    ppl = stderr = kl = None
    if m := _PPL_RE.search(output):
        ppl, stderr = float(m.group(1)), float(m.group(2))
    if m := _KL_RE.search(output):
        kl = float(m.group(1))

    result = {
        "label": args.label,
        "base": str(args.base),
        "target": str(args.target),
        "eval": str(args.eval),
        "eval_sha256": _sha256(args.eval),
        "timestamp": datetime.now(UTC).isoformat(),
        "binary": binary,
        "command": command,
        "command_str": shlex.join(command),
        "returncode": proc.returncode,
        "elapsed_s": dt,
        "ppl": ppl,
        "ppl_stderr": stderr,
        "kl_mean": kl,
        "output_tail": output[-4000:],
    }

    out = args.out_dir / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if result["returncode"] != 0 or kl is None:
        print(f"rc={proc.returncode} kl={kl}  -> see {out}")
        return 1
    print(f"{args.label}: KL {kl:.6f}  PPL {ppl:.4f}  ({dt:.1f}s)  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
