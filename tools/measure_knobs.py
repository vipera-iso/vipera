#!/usr/bin/env python3
"""One-off A/B of llama.cpp load knobs on the real card: decode t/s + TTFT.

Loads `llama_cpp.Llama` with exactly the flags `Engine._load` uses, plus the
knob under test, so a row is directly comparable to `vipera bench run`. Every
row is a fresh load; decode t/s is measured after the first streamed token.

usage: measure_knobs.py <model-alias> '[["label",{"threads":4,"ngl":12}], ...]'
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from vipera.hardware import detect
from vipera.planner import plan
from vipera.registry import get_model

PROMPT = (
    "Write a highly detailed and comprehensive guide on how to build a modern "
    "web application using Python. Include sections on backend architecture, "
    "database design, API routing, and frontend integration. Make sure to explain "
    "the concepts clearly for a beginner."
)

_KV = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
    "iq4_nl": 20,
    "bf16": 30,
}

MODEL = sys.argv[1] if len(sys.argv) > 1 else "NousResearch_Hermes-4-14B-Q4_K_M-2"
CASES = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [["control", {}]]
MAX_TOKENS = 64
REPS = int(os.environ.get("REPS", "3"))


def _vram_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _resolve(model):
    entry = get_model(model)
    if entry:
        return entry["path"], entry.get("meta", {})
    path = Path(model).expanduser()
    from vipera.gguf_analyzer import analyze_gguf

    meta = analyze_gguf(str(path))["metadata"]
    meta["size_bytes"] = path.stat().st_size
    return str(path), meta


def one(over):
    from llama_cpp import Llama

    path, meta = _resolve(MODEL)
    entry = {"path": path, "meta": meta}
    hw = detect()
    p = plan(
        entry["meta"],
        hw,
        ctx=over.get("ctx", 8192),
        kv_cache=over.get("kv", "q8_0"),
        n_batch=over.get("n_batch", 512),
    )
    ngl = over.get("ngl", p.gpu_layers)
    threads = over.get("threads", p.cpu_threads)
    kv = _KV.get(over.get("kv", p.kv_cache))
    fa = over.get("fa", p.flash_attn)
    t0 = time.perf_counter()
    try:
        llm = Llama(
            model_path=entry["path"],
            n_gpu_layers=ngl,
            n_ctx=over.get("ctx", 8192),
            n_threads=threads,
            n_threads_batch=over.get("threads_batch") or None,
            n_batch=over.get("ubatch", over.get("n_batch", 512)),
            n_ubatch=over.get("ubatch", 512),
            flash_attn=fa,
            offload_kqv=p.offload_kqv,
            type_k=kv,
            type_v=kv,
            use_mmap=over.get("use_mmap", True),
            use_mlock=over.get("mlock", p.use_mlock),
            op_offload=over.get("op_offload", True),
            verbose=False,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:160], "ngl": ngl}
    load_s = time.perf_counter() - t0
    vram_after_load = _vram_mib()
    prompt = over.get("prompt") or PROMPT
    prompt_tok = len(llm.tokenize(prompt.encode()))
    stream = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    first = None
    n = 0
    t0 = time.perf_counter()
    for _ in stream:
        if first is None:
            first = time.perf_counter()
        n += 1
    done = time.perf_counter()
    del llm
    dec_s = done - first
    return {
        "planned_ngl": p.gpu_layers,
        "ngl": ngl,
        "threads": threads,
        "kv": over.get("kv", p.kv_cache),
        "load_s": round(load_s, 2),
        "ttft_s": round(first - t0, 3),
        "prompt_tok": prompt_tok,
        "prefill_tps": round(prompt_tok / max(first - t0, 1e-9), 1),
        "tok": n,
        "decode_tps": round((n - 1) / dec_s, 3) if dec_s > 0 else 0,
        "vram_load_mib": vram_after_load,
        "vram_end_mib": _vram_mib(),
    }, p


def main():
    import gc

    hw = detect()
    print(
        f"# gpu={hw.gpu_name} vram_free={hw.vram_free_gb:.2f} cores={hw.cpu_cores}", file=sys.stderr
    )
    for label, over in CASES:
        rows = []
        for _ in range(REPS):
            out = one(over)
            gc.collect()
            time.sleep(2)
            if isinstance(out, tuple):
                rows.append(out[0])
        if not rows or "error" in rows[0]:
            print(
                json.dumps(
                    {
                        "label": label,
                        "over": over,
                        "error": rows[0].get("error") if rows else "failed",
                    }
                ),
                flush=True,
            )
            continue
        tps = [r["decode_tps"] for r in rows]
        print(
            json.dumps(
                {
                    "label": label,
                    "over": {k: v for k, v in over.items() if k != "prompt"},
                    "ngl": rows[-1]["ngl"],
                    "vram_mib": rows[-1]["vram_load_mib"],
                    "threads": rows[-1]["threads"],
                    "median_tps": round(statistics.median(tps), 3),
                    "runs_tps": tps,
                    "median_ttft_s": round(statistics.median([r["ttft_s"] for r in rows]), 3),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
