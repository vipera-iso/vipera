#!/usr/bin/env python3
"""A/B the placements a spill MoE model can be run with, on this machine.

A model larger than RAM has no "correct" llama.cpp command line to copy: the
planner picks one from the hardware, and the only way to know whether it picked
right is to run both and compare. This harness does exactly that, on the model
and the machine in `~/.vipera/config.json`, one arm at a time.

Arms
----
`classic`    what the planner produced before the shard fix: whole layers on the
             card, expert slabs included (`-ngl 2` on a 36-layer gpt-oss-120b).
`attention`  the plan for a MoE model whose expert pool cannot be placed:
             every routed expert on the CPU, the always-active side on the card
             (`-ngl 37 -ncmoe 36`).
`+flags`     any arm can carry extra llama-server flags, e.g.
             `--extra "--moe-cache-mib 1024"`.

Every arm replays Vipera's own `Plan` through `backends.server.server_command`,
so the flags under test are the flags the gateway would ship.

What is measured
----------------
* **gen t/s** — median of N requests, first token to last.
* **TTFT** — includes prefill and, for the first request, the page faults that
  read the model off disk.
* **read_bytes per token** — bytes the child actually read from storage during
  the measured window (`/proc/<pid>/io`). This is the mechanism number: a token
  that costs 0.8 GiB of storage reads at 1 t/s is bound by the disk, whatever
  the tok/s says about compute.
* **major faults per token** — useful, but **do not read 0.00 as "no disk"**: the
  kernel's readahead often brings the page in before the process faults, so the
  fault is minor and the counter stays at zero while read_bytes climbs. Measured
  on this box, all arms read 0.83-1.02 GiB/token from storage at 0.00 majflt.
* **peak RSS**, **VRAM used** — what the placement costs.

Usage
-----
    python tools/sweep_spill_moe.py                       # every arm below
    python tools/sweep_spill_moe.py --arms attention
    python tools/sweep_spill_moe.py --ctx 4096 --tokens 32 --repeats 3

Results go to `tools/spill_results/<alias>-<timestamp>.json`, raw argv included,
so a number can be reproduced by copying one line out of the file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vipera.backends import server as server_backend
from vipera.config import load_config
from vipera.hardware import detect
from vipera.planner import Plan, plan
from vipera.registry import get_model

SERVER = "/home/user/.vipera/bin/llama-server"
PORT = 8791
PROMPT = (
    "Write a short Python function that reads a GGUF file's header and returns "
    "the tensor names and shapes. Then explain in three sentences how a MoE "
    "router decides which experts run for a token, and why the expert weights "
    "dominate the bytes read per token. Be concrete and avoid filler."
)

# What the planner saw before the shard fix, kept here as the control arm: the
# whole-file size from the registry against shard 1's tables. Reproduced rather
# than hand-written so the classic arm is a real planner output, not a guess.
STALE_META = {
    "n_layers": 36,
    "n_heads": 64,
    "n_kv_heads": 8,
    "head_dim": 64,
    "kv_layers": 36,
    "context_length": 131072,
    "size_bytes": 62570256992,
    "expert_bytes": 48068444160,
    "expert_layers": 29,
    "expert_layer_bytes": [1696481280] * 28 + [566968320] + [0] * 7,
    "attn_bytes": 48504216320,
    "fixed_bytes": 941103360,
}


@dataclass(frozen=True)
class Arm:
    name: str
    note: str
    plan: Plan
    extra: list[str]


def build_arms(model, cfg, hw, *, ctx: int) -> list[Arm]:
    """Every placement worth measuring, all from Vipera's own planner."""
    current = plan(
        model.meta,
        hw,
        profile=cfg.get("profile", "auto"),
        ctx=ctx,
        kv_cache=cfg.get("kv_cache", "q8_0"),
        n_batch=cfg.get("n_batch", 512),
        n_ubatch=cfg.get("n_ubatch", 0),
    )
    classic = plan(
        STALE_META,
        hw,
        profile=cfg.get("profile", "auto"),
        ctx=ctx,
        kv_cache=cfg.get("kv_cache", "q8_0"),
        n_batch=cfg.get("n_batch", 512),
        n_ubatch=cfg.get("n_ubatch", 0),
    )
    return [
        Arm("classic", classic.note, classic, []),
        Arm("attention", current.note, current, []),
        Arm(
            "attention+moe-pool",
            current.note + " + --moe-cache-mib 1024",
            current,
            ["--moe-cache-mib", "1024"],
        ),
        Arm(
            "classic+ngram",
            classic.note + " + --spec-type ngram-map-k",
            classic,
            ["--spec-type", "ngram-map-k"],
        ),
    ]


def _peak_rss_bytes(pid: int) -> int:
    """VmHWM in bytes, or 0 when the process is already gone."""
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _proc_io(pid: int) -> tuple[int, int]:
    """(read_bytes, major faults) for a process, best effort."""
    read_bytes = major = 0
    try:
        with open(f"/proc/{pid}/io") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    read_bytes = int(line.split()[1])
                elif line.startswith("majflt:"):
                    major = int(line.split()[1])
    except OSError:
        pass
    return read_bytes, major


def _vram_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _wait_health(port: int, timeout: float) -> float:
    """Seconds until the child reports ready; 503 {"status": "loading"} keeps waiting."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200 and json.loads(resp.read()).get("status") == "ok":
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"no ready /health on port {port} after {timeout:.0f}s")


def _chat(port: int, max_tokens: int) -> dict:
    """One streaming completion; returns TTFT, generation time and token count."""
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(request, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or {}
            if usage:
                # llama.cpp only reports usage when asked, and it arrives last.
                tokens = usage.get("completion_tokens") or tokens
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            # gpt-oss is a reasoning model: with a small max_tokens every token
            # it emits lands in `reasoning_content`, and waiting for `content`
            # would time the whole run at "no output".
            if first is None and any(
                delta.get(field) for field in ("content", "reasoning_content", "reasoning")
            ):
                first = time.monotonic()
    finished = time.monotonic()
    if first is None:
        raise RuntimeError("no content in the stream")
    return {
        "ttft_s": first - started,
        "gen_s": finished - first,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
    }


def _chat_ready(port: int, max_tokens: int, attempts: int = 10) -> dict:
    """`_chat`, retrying while the server is still warming up (503)."""
    for attempt in range(attempts):
        try:
            return _chat(port, max_tokens)
        except urllib.error.HTTPError as exc:
            if exc.code not in (500, 503) or attempt == attempts - 1:
                raise
            time.sleep(2.0)
    raise RuntimeError("unreachable")


def run_arm(arm: Arm, model_path: str, *, ctx: int, tokens: int, repeats: int) -> dict:
    command = [
        *server_backend.server_command(
            SERVER, model_path, arm.plan, PORT, moe_cache_supported=True, spec_type=None
        ),
        *arm.extra,
    ]
    print(f"\n=== {arm.name}: {arm.note}")
    print("    " + " ".join(command))
    log = Path("/tmp") / f"sweep-spill-{arm.name.replace('+', '_')}.log"
    with log.open("wb") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    try:
        load_s = _wait_health(PORT, timeout=900)
        pid = child.pid
        first = _chat_ready(PORT, tokens)  # pays the page faults; not a measurement
        runs = []
        io_before = _proc_io(pid)
        for _ in range(repeats):
            runs.append(_chat_ready(PORT, tokens))
        io_after = _proc_io(pid)
        rss = _peak_rss_bytes(pid)
        generated = sum(run["tokens"] for run in runs)
        tps = [run["tokens"] / run["gen_s"] for run in runs]
        prefill_tps = statistics.median((run["prompt_tokens"] or 1) / run["ttft_s"] for run in runs)
        result = {
            "arm": arm.name,
            "note": arm.note,
            "argv": command,
            "load_s": round(load_s, 1),
            "warmup_ttft_s": round(first["ttft_s"], 2),
            "runs": [
                {
                    "ttft_s": round(run["ttft_s"], 2),
                    "gen_s": round(run["gen_s"], 2),
                    "tokens": run["tokens"],
                }
                for run in runs
            ],
            "gen_tps_median": round(statistics.median(tps), 3),
            "gen_tps_all": [round(value, 3) for value in tps],
            "prefill_tps_median": round(prefill_tps, 2),
            "read_gib_per_token": round(
                (io_after[0] - io_before[0]) / 1024**3 / max(1, generated), 4
            ),
            "major_faults_per_token": round((io_after[1] - io_before[1]) / max(1, generated), 2),
            "peak_rss_gib": round(rss / 1024**3, 2),
            "vram_used_mib": _vram_used_mib(),
            "log": str(log),
        }
        print(
            f"    gen {result['gen_tps_median']} t/s | prefill {result['prefill_tps_median']} t/s"
            f" | {result['read_gib_per_token']} GiB read/token"
            f" | {result['major_faults_per_token']} majflt/token"
            f" | RSS {result['peak_rss_gib']} GiB | VRAM {result['vram_used_mib']} MiB"
        )
        return result
    finally:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=None, help="registry alias (default: config default_model)"
    )
    parser.add_argument("--ctx", type=int, default=None, help="context (default: config)")
    parser.add_argument("--tokens", type=int, default=32, help="tokens per measured request")
    parser.add_argument("--repeats", type=int, default=3, help="measured requests per arm")
    parser.add_argument("--arms", default="", help="comma-separated subset of arm names")
    parser.add_argument("--out", default="tools/spill_results", help="output directory")
    parser.add_argument("--print-arms", action="store_true", help="print each arm's argv and exit")
    args = parser.parse_args()

    cfg = load_config()
    alias = args.model or cfg.get("default_model")
    model = get_model(alias)
    if model is None:
        print(f"no model called {alias} in the registry", file=sys.stderr)
        return 1
    hw = detect()
    ctx = args.ctx or int(cfg.get("ctx") or 8192)

    arms = build_arms(model, cfg, hw, ctx=ctx)
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",") if name.strip()}
        arms = [arm for arm in arms if arm.name in wanted]
        if not arms:
            print(f"no arm matched {args.arms!r}", file=sys.stderr)
            return 1

    print(f"model  {alias} ({model.path.name})")
    print(f"ctx    {ctx}   tokens/run {args.tokens}   repeats {args.repeats}")
    print(f"card   {hw.gpu_name} {hw.vram_free_gb:.2f} GB free of {hw.vram_total_gb:.2f} GB")
    print(f"ram    {hw.ram_free_gb:.2f} GB free of {hw.ram_total_gb:.2f} GB")

    if args.print_arms:
        for arm in arms:
            command = server_backend.server_command(
                SERVER, str(model.path), arm.plan, PORT, moe_cache_supported=True, spec_type=None
            )
            print(f"\n{arm.name}: {arm.note}")
            print("    " + " ".join([*command, *arm.extra]))
        return 0

    results = {
        "alias": alias,
        "model_path": str(model.path),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "ctx": ctx,
        "tokens_per_run": args.tokens,
        "repeats": args.repeats,
        "hardware": {
            "gpu": hw.gpu_name,
            "vram_total_gb": hw.vram_total_gb,
            "vram_free_gb": hw.vram_free_gb,
            "ram_total_gb": hw.ram_total_gb,
            "ram_free_gb": hw.ram_free_gb,
            "cpu_cores": hw.cpu_cores,
        },
        "arms": [],
    }
    for arm in arms:
        try:
            results["arms"].append(
                run_arm(arm, str(model.path), ctx=ctx, tokens=args.tokens, repeats=args.repeats)
            )
        except Exception as exc:  # one arm failing must not lose the others
            print(f"    arm failed: {exc}")
            results["arms"].append({"arm": arm.name, "note": arm.note, "error": str(exc)})
        Path(args.out).mkdir(parents=True, exist_ok=True)
        out = Path(args.out) / f"{alias}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
        out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
