#!/usr/bin/env python3
"""Run `llama-perplexity` (llama.cpp) on one GGUF and record PPL + metadata (L5a).

Each run writes one JSON so a delta can be recomputed without re-running.
`compare` prints delta + delta/stderr, the only honest way to say "improved"
vs "within noise".

Uses find_llama_tool("perplexity"); needs the "perplexity" alias in
optimizer._TOOL_NAMES (see the companion optimizer patch).

    python tools/l5a_perplexity.py run --model a.gguf --eval eval.txt --label control
    python tools/l5a_perplexity.py run --model b.gguf --eval eval.txt --label variant
    python tools/l5a_perplexity.py compare \\
        tools/l5a_results/control.json tools/l5a_results/variant.json

--resume skips a run whose output JSON exists with returncode 0, matching the
resume behaviour of tools/l1a2_run.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _run(
    binary: str,
    model: Path,
    eval_file: Path,
    ctx: int,
    chunks: int,
    ngl: int,
    n_seq: int,
    extra: list[str],
) -> dict:
    command = [
        binary,
        "-m",
        str(model),
        "-f",
        str(eval_file),
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "-ngl",
        str(ngl),
    ]
    if n_seq > 1:
        command.extend(["-n-seq", str(n_seq)])
    command.extend(extra)

    t0 = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    output = f"{proc.stdout}\n{proc.stderr}"

    ppl = stderr = kl = None
    if m := _PPL_RE.search(output):
        ppl, stderr = float(m.group(1)), float(m.group(2))
    if m := _KL_RE.search(output):
        kl = float(m.group(1))

    return {
        "command": command,
        "command_str": shlex.join(command),
        "returncode": proc.returncode,
        "elapsed_s": dt,
        "ppl": ppl,
        "ppl_stderr": stderr,
        "kl_mean": kl,
        "output_tail": output[-4000:],
    }


def _compare(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    if a["ppl"] is None or b["ppl"] is None:
        print("cannot compare: one run has no PPL (rc != 0?)")
        return 1

    delta = b["ppl"] - a["ppl"]
    pct = delta / a["ppl"] * 100.0
    # Conservative noise floor: same eval set => runs are correlated, so a
    # sqrt(2)*stderr floor would OVERSTATE noise. Using max(stderr) errs on
    # the side of "cannot conclude", which is the safe direction.
    sigma = max(a["ppl_stderr"] or 0.0, b["ppl_stderr"] or 0.0)
    ratio = abs(delta) / sigma if sigma > 0 else (math.inf if delta else 0.0)

    print(f"a: {a_path}  PPL {a['ppl']:.4f} +/- {a['ppl_stderr'] or 0:.4f}")
    print(f"b: {b_path}  PPL {b['ppl']:.4f} +/- {b['ppl_stderr'] or 0:.4f}")
    print(f"delta: {delta:+.4f}  ({pct:+.2f}%)  = {ratio:.2f} sigma")
    if abs(delta) < sigma:
        print("verdict: within noise; cannot conclude a difference")
    elif delta < 0:
        print("verdict: b is better (lower PPL) beyond noise")
    else:
        print("verdict: b is worse (higher PPL) beyond noise")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run llama-perplexity on one model")
    run_p.add_argument("--model", required=True, type=Path)
    run_p.add_argument("--eval", required=True, type=Path)
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--ctx", type=int, default=512)
    run_p.add_argument("--chunks", type=int, default=24)
    run_p.add_argument("--ngl", type=int, default=0)
    run_p.add_argument("--n-seq", type=int, default=4, dest="n_seq")
    run_p.add_argument("--out-dir", type=Path, default=Path("tools/l5a_results"))
    run_p.add_argument("--bin", help="path to llama-perplexity (default: find it)")
    run_p.add_argument(
        "--resume", action="store_true", help="skip if the output JSON already has rc=0"
    )
    run_p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    cmp_p = sub.add_parser("compare", help="compare two run JSONs")
    cmp_p.add_argument("a", type=Path)
    cmp_p.add_argument("b", type=Path)

    args = parser.parse_args()
    if args.cmd == "compare":
        return _compare(args.a, args.b)

    out = args.out_dir / f"{args.label}.json"
    if args.resume and out.exists():
        try:
            prev = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError):
            prev = None
        if prev and prev.get("returncode") == 0 and prev.get("ppl") is not None:
            print(f"{args.label}: resume, using cached {out}")
            return 0

    binary = args.bin
    if not binary:
        try:
            binary = find_llama_tool("perplexity")
        except OptimizerError as exc:
            print(f"{exc}")
            return 1

    result = _run(
        binary,
        args.model,
        args.eval,
        args.ctx,
        args.chunks,
        args.ngl,
        args.n_seq,
        args.extra or [],
    )
    result.update(
        {
            "label": args.label,
            "model": str(args.model),
            "eval": str(args.eval),
            "eval_sha256": _sha256(args.eval),
            "timestamp": datetime.now(UTC).isoformat(),
            "binary": binary,
            "ctx": args.ctx,
            "chunks": args.chunks,
            "ngl": args.ngl,
            "n_seq": args.n_seq,
        }
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if result["returncode"] != 0 or result["ppl"] is None:
        print(f"rc={result['returncode']} ppl={result['ppl']}  -> see {out}")
        return 1
    print(
        f"{args.label}: PPL {result['ppl']:.4f} +/- {result['ppl_stderr'] or 0:.4f}  "
        f"({result['elapsed_s']:.1f}s)  -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
