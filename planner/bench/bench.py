"""Run llama-bench and record timing *and* resource cost.

llama-bench already reports prefill and decode t/s. What it does not report is
peak RSS, peak VRAM and disk I/O — the numbers that actually decide whether a model
runs on a weak machine. This wraps it, samples the process, and emits one JSON report.

    python3 -m bench.bench --bin ~/dev/llama.cpp/build/bin/llama-bench \
        --model models/deepseek-v2-lite-Q4_K_M.gguf --ngl 0 --json bench/results/baseline.json

Exit codes: 0 measured, 1 the benchmark failed, 2 bad usage or unreadable JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

try:  # runnable both as `python3 -m bench.bench` and as a script
    from .measure import Sampler, bytes_per_token
except ImportError:  # pragma: no cover
    from measure import Sampler, bytes_per_token  # type: ignore

DEFAULT_BIN = "llama-bench"


def build_command(binary: str, model: str, *, ngl: int | None, threads: int | None,
                  pp: int, tg: int, extra: list[str]) -> list[str]:
    cmd = [binary, "-m", model, "-o", "json", "-p", str(pp), "-n", str(tg)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    if threads is not None:
        cmd += ["-t", str(threads)]
    cmd += extra
    return cmd


def parse_llama_bench_json(text: str) -> list[dict]:
    """llama-bench may print a JSON array, or an object with a 'results' array."""
    data = json.loads(text)
    if isinstance(data, dict):
        # An object without 'results' must be an error, not an empty run: silently
        # returning [] would let a malformed run be reported as a clean pass.
        if "results" not in data:
            raise ValueError("llama-bench JSON object has no 'results' array")
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError("llama-bench JSON is neither a list nor an object with 'results'")
    return data


def row_label(r: dict) -> str:
    """Human label for one llama-bench row.

    llama-bench's JSON has no `test` field (only the markdown table does), so the
    label is rebuilt from n_prompt/n_gen exactly the way llama-bench names them.
    """
    if r.get("test"):
        return str(r["test"])
    p = int(r.get("n_prompt") or 0)
    g = int(r.get("n_gen") or 0)
    if p and not g:
        return f"pp{p}"
    if g and not p:
        return f"tg{g}"
    if p or g:
        return f"pp{p}+tg{g}"
    return "unknown"


def total_tokens(results: list[dict]) -> int:
    n = 0
    for r in results:
        n += int(r.get("n_prompt") or 0) + int(r.get("n_gen") or 0)
    return n


def run_bench(binary: str, model: str, *, ngl: int | None = None, threads: int | None = None,
              pp: int = 512, tg: int = 128, extra: list[str] | None = None,
              sample_interval: float = 0.2, gpu: bool = True,
              timeout: float | None = None) -> dict:
    cmd = build_command(binary, model, ngl=ngl, threads=threads, pp=pp, tg=tg, extra=extra or [])

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = Sampler(proc.pid, interval=sample_interval, gpu=gpu).start()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        peak = sampler.stop()
        return {
            "ok": False,
            "error": "timeout",
            "started_at": started_at,
            "elapsed_s": time.monotonic() - started,
            "command": cmd,
            "peak_rss_kb": peak.rss_kb,
            "peak_gpu_mib": peak.gpu_mib,
            "samples": sampler.samples,
        }
    peak = sampler.stop()
    elapsed = time.monotonic() - started

    report: dict = {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "started_at": started_at,
        "elapsed_s": round(elapsed, 3),
        "command": cmd,
        "peak_rss_kb": peak.rss_kb,
        "peak_gpu_mib": peak.gpu_mib,
        "samples": sampler.samples,
    }

    if peak.io:
        report["io"] = {
            "read_bytes": peak.io.get("read_bytes"),
            "write_bytes": peak.io.get("write_bytes"),
        }

    if proc.returncode != 0:
        report["stderr_tail"] = err[-2000:]
        return report

    try:
        results = parse_llama_bench_json(out)
    except (ValueError, json.JSONDecodeError) as e:
        report["ok"] = False
        report["error"] = f"could not parse llama-bench JSON: {e}"
        report["stdout_tail"] = out[-2000:]
        return report

    tokens = total_tokens(results)
    read_bytes = report.get("io", {}).get("read_bytes")
    report["results"] = [
        {
            "test": row_label(r),
            "n_prompt": r.get("n_prompt"),
            "n_gen": r.get("n_gen"),
            "avg_ts": r.get("avg_ts"),
            "stddev_ts": r.get("stddev_ts"),
        }
        for r in results
    ]
    report["tokens_processed"] = tokens
    report["bytes_read_per_token"] = bytes_per_token(read_bytes, tokens)
    report["model_filename"] = results[0].get("model_filename") if results else None
    return report


def format_table(report: dict) -> str:
    lines = []
    lines.append(f"ok            {report.get('ok')}")
    if "exit_code" in report:
        lines.append(f"exit          {report['exit_code']}")
    lines.append(f"elapsed       {report.get('elapsed_s')} s")
    if report.get("peak_rss_kb") is not None:
        lines.append(f"peak RSS      {report['peak_rss_kb'] / 1024:.1f} MiB")
    if report.get("peak_gpu_mib") is not None:
        lines.append(f"peak VRAM     {report['peak_gpu_mib']} MiB")
    io = report.get("io")
    if io and io.get("read_bytes") is not None:
        lines.append(f"disk read     {io['read_bytes'] / (1024 * 1024):.1f} MiB")
    if report.get("bytes_read_per_token") is not None:
        lines.append(f"read/token    {report['bytes_read_per_token'] / (1024 * 1024):.3f} MiB")
    for r in report.get("results", []):
        ts = r.get("avg_ts")
        ts_text = f"{ts:.2f} t/s" if isinstance(ts, (int, float)) else "n/a"
        lines.append(f"  {str(r.get('test') or 'unknown'):<12} {ts_text}")
    if report.get("error"):
        lines.append(f"error         {report['error']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench", description=__doc__)
    p.add_argument("--bin", default=DEFAULT_BIN, help="llama-bench executable")
    p.add_argument("--model", required=True)
    p.add_argument("--ngl", type=int, default=None, help="layers offloaded to GPU")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--pp", type=int, default=512, help="prompt tokens per test")
    p.add_argument("--tg", type=int, default=128, help="generated tokens per test")
    p.add_argument("--interval", type=float, default=0.2, help="sampling interval (s)")
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--no-gpu", action="store_true", help="skip nvidia-smi sampling")
    p.add_argument("--json", dest="json_path", help="write the report here")
    return p


def split_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse bench's own flags and hand the rest through to llama-bench.

    Unknown flags (e.g. `-r 1 --no-warmup`) are returned verbatim and in order, so
    a documented llama-bench flag no longer has to be hidden behind a `--`
    separator. `parse_known_args` keeps a stray positional in the same list.
    """
    args, extra = build_parser().parse_known_args(argv)
    # argparse hands the `--` separator itself back in extras; it is bench's own
    # delimiter, not a llama-bench flag, so drop it.
    extra = [a for a in extra if a != "--"]
    return args, extra


def main(argv: list[str] | None = None) -> int:
    args, extra = split_args(argv)

    binary = shutil.which(args.bin) or args.bin
    report = run_bench(
        binary, args.model,
        ngl=args.ngl, threads=args.threads, pp=args.pp, tg=args.tg,
        extra=extra, sample_interval=args.interval, gpu=not args.no_gpu,
        timeout=args.timeout,
    )
    print(format_table(report))
    if args.json_path:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"report written to {args.json_path}")
    if report.get("error", "").startswith("could not parse"):
        return 2
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
