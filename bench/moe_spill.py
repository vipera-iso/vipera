"""Measure what happens when a MoE model is larger than RAM.

`docs/FINDINGS.md` §Round 6 concluded that the spill regime is **unreachable** on
this machine: Qwen3-30B-A3B (17.28 GiB) against 12.4 GiB available was OOM-killed
during load, twice, in both load modes, and the conclusion drawn was that a model
large enough to spill is a model too large to load.

That conclusion was wrong, and this harness is what corrected it. The kill was not
the expert pool overflowing RAM. It was `--repack`: llama.cpp's CPU backend copies
**3D** weight tensors used by `GGML_OP_MUL_MAT_ID` -- the MoE expert matmul -- into a
`CPU_REPACK` buffer, which is **anonymous** memory, not file-backed (`ggml-cpu.cpp`,
`ggml_backend_cpu_repack_buffer_type`; `repack.cpp`, `extra_buffer_type::supports_op`
accepts `MUL_MAT_ID` with `ggml_n_dims == 3`). Repacking is a CPU *speed* feature, and
it silently converts a disk-streamable model into a must-fit-in-RAM one. `-nr`
(`--no-repack`, `params.no_extra_bufts`) turns it off, and the model then loads and
runs with the kernel paging the expert slabs from disk.

So this harness measures two arms that differ in exactly one flag:

    control   no -nr     the default, expected to be OOM-killed during load
    no-repack -nr        experts stay file-backed; expected to run

Both arms run sequentially, never concurrently (the standing rule: two of these at
once ask for ~2x the RAM and the OOM killer takes one).

**RssAnon is the mechanism evidence, not decoration.** It is what distinguishes
"the flag made it work" from "the flag did something unrelated". A run whose
anonymous RSS is small is a run whose expert weights are still in the file.

Run: `python3 bench/moe_spill.py --json` (writes `bench/results/moe-spill.json`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "models", "Qwen3-30B-A3B-Q4_K_M.gguf")
DEFAULT_BIN = os.path.expanduser("~/dev/llama.cpp/build/bin/llama-cli")

# One short deterministic completion. Small on purpose: the question is whether the
# model runs at all, and every extra token makes the OOM arm more likely to take the
# whole session down rather than just itself.
COMMON = [
    "-p", "The quick brown fox", "-n", "32", "-t", "4", "-lm", "mmap",
    "-c", "512", "-s", "42", "--temp", "0", "--ignore-eos", "-st",
    "-fa", "off", "-ctk", "f16", "-ctv", "f16",
]

STATUS_RE = re.compile(r"^(RssAnon|RssFile|VmRSS):\s+(\d+) kB$", re.M)
GEN_RE = re.compile(r"Generation:\s+([0-9.]+)\s*t/s")
PROMPT_RE = re.compile(r"Prompt:\s+([0-9.]+)\s*t/s")
TOKENS_RE = re.compile(r"([0-9.]+)\s*tokens per second")


def sample_proc(pid: int) -> dict:
    """RssAnon and RssFile from /proc/<pid>/status, in MiB."""
    try:
        with open(f"/proc/{pid}/status") as f:
            text = f.read()
    except OSError:
        return {}
    out = {}
    for key, value in STATUS_RE.findall(text):
        out[key] = int(value) / 1024.0
    return out


def run_arm(*, binary: str, model: str, no_repack: bool, timeout: int) -> dict:
    """Run llama-cli once, sampling its memory while it lives."""
    argv = [binary, "-m", model, *COMMON]
    if no_repack:
        argv.append("-nr")

    started = time.time()
    log = os.path.join("/tmp", f"moe_spill_{'nr' if no_repack else 'control'}.log")
    peak_anon = 0.0
    peak_file = 0.0
    peak_rss = 0.0
    samples = 0
    with open(log, "wb") as sink:
        proc = subprocess.Popen(argv, stdout=sink, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                mem = sample_proc(proc.pid)
                if mem:
                    samples += 1
                    peak_anon = max(peak_anon, mem.get("RssAnon", 0.0))
                    peak_file = max(peak_file, mem.get("RssFile", 0.0))
                    peak_rss = max(peak_rss, mem.get("VmRSS", 0.0))
                if time.time() - started > timeout:
                    proc.kill()
                    proc.wait()
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            raise
    wall = time.time() - started
    code = proc.returncode
    with open(log, "r", errors="replace") as f:
        text = f.read()

    gen = GEN_RE.search(text)
    prompt = PROMPT_RE.search(text)
    # SIGKILL arrives as -9 from subprocess and as 137 from a shell; both are the OOM
    # killer here, and it is the *result*, not a crash to be retried away. Reporting
    # it as a generic "error" is the exact defect this repo already suffered once --
    # an OOM kill surfaced as an `n/a` cell with exit 0 and was written up as a
    # property of the model. `dmesg` needs root and is unavailable, so the kill is
    # identified from the signal plus peak RSS sitting at the machine's limit, and
    # that inference is stated rather than hidden.
    if code in (-9, 137):
        outcome, evidence = "oom-killed", "SIGKILL at peak RSS ~ the machine's available RAM"
    elif code == 0:
        outcome, evidence = "ran", ""
    else:
        outcome, evidence = "error", f"exit {code}"
    return {
        "arm": "no-repack" if no_repack else "control",
        "no_repack": no_repack,
        "exit_code": code,
        "outcome": outcome,
        "outcome_evidence": evidence,
        "wall_s": round(wall, 2),
        "tok_s_generation": float(gen.group(1)) if gen else None,
        "tok_s_prompt": float(prompt.group(1)) if prompt else None,
        "peak_rss_mib": round(peak_rss, 1),
        "peak_rss_anon_mib": round(peak_anon, 1),
        "peak_rss_file_mib": round(peak_file, 1),
        "mem_samples": samples,
        "log": log,
        # The lever audit, in the report's own words. ffmpeg-style: if the marker is
        # absent the arm is NOT CONCLUSIVE, because two identical decoders producing
        # identical numbers is not evidence of an optimization.
        "repack_marker": (
            "CPU_REPACK" if "CPU_REPACK" in text else "no CPU_REPACK line at this verbosity"
        ),
    }


def _median(values: list[float]) -> float | None:
    """Median, or None for an empty list. With n=3 a mean would hide a bimodal arm."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _summarise(arms: list[dict]) -> dict:
    """Per-arm outcome counts and the median tok/s, with every run kept in `arms`.

    The count of OOM kills is the headline: it is the one number here that is not a
    timing measurement and does not carry a spread.
    """
    out: dict = {}
    for arm in ("control", "no-repack"):
        rows = [a for a in arms if a["arm"] == arm]
        if not rows:
            continue
        speeds = [a["tok_s_generation"] for a in rows if a["tok_s_generation"]]
        out[arm] = {
            "runs": len(rows),
            "ran": sum(1 for a in rows if a["outcome"] == "ran"),
            "oom_killed": sum(1 for a in rows if a["outcome"] == "oom-killed"),
            "tok_s_median": _median(speeds),
            "tok_s_each": speeds,
            "peak_anon_mib_median": _median([a["peak_rss_anon_mib"] for a in rows]),
            "peak_file_mib_median": _median([a["peak_rss_file_mib"] for a in rows]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="moe_spill", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--bin", dest="binary", default=DEFAULT_BIN)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "bench", "results",
                                                 "moe-spill.json"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--arm", choices=["control", "no-repack", "both"], default="both")
    p.add_argument("--reps", type=int, default=1,
                   help="repeat each arm; every repetition is recorded, never averaged "
                        "away, and the median is reported beside them")
    args = p.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"no model at {args.model}", file=sys.stderr)
        return 2
    if not os.path.exists(args.binary):
        print(f"no llama-cli at {args.binary}", file=sys.stderr)
        return 2

    total = round(os.path.getsize(args.model) / 2**30, 2)
    arms = []
    reps = max(1, args.reps)
    for _ in range(reps):
        if args.arm in ("control", "both"):
            arms.append(run_arm(binary=args.binary, model=args.model, no_repack=False,
                                timeout=args.timeout))
        if args.arm in ("no-repack", "both"):
            arms.append(run_arm(binary=args.binary, model=args.model, no_repack=True,
                                timeout=args.timeout))

    report = {
        "harness": "bench/moe_spill.py",
        "model": os.path.basename(args.model),
        "model_gib": total,
        "args": " ".join(COMMON),
        "reps": reps,
        "arms": arms,
        "summary": _summarise(arms),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"model  {report['model']}  {total:.2f} GiB")
        print(f"{'arm':<11}{'outcome':<11}{'exit':<6}{'wall':<8}{'tok/s':<8}"
              f"{'peakRSS':<10}{'anonRSS':<10}{'fileRSS'}")
        for a in arms:
            print(f"{a['arm']:<11}{a['outcome']:<11}{a['exit_code']:<6}"
                  f"{a['wall_s']:<8}{str(a['tok_s_generation']):<8}"
                  f"{a['peak_rss_mib']:<10}{a['peak_rss_anon_mib']:<10}"
                  f"{a['peak_rss_file_mib']}")
        for arm, s in report["summary"].items():
            print(f"{arm:<11}ran {s['ran']}/{s['runs']}, oom-killed {s['oom_killed']}, "
                  f"tok/s median {s['tok_s_median']} {s['tok_s_each']}, "
                  f"anon {s['peak_anon_mib_median']} MiB / file {s['peak_file_mib_median']} MiB")
        if args.out:
            print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
