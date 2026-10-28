#!/usr/bin/env python3
"""STEP 1 bottleneck attribution (plantodo.md section 1).

Splits the decode time of a CPU-only MoE run (DeepSeek-V2-Lite Q4_K_M) into
disk / RAM / compute, without perf(1): perf_event_paranoid=2 and no sudo on the
target machine, so /proc counters + llama-bench's own timing are used instead.

Levers and what each separates:
  -t 1,2,4,8       thread scaling: ALU-bound keeps scaling with threads,
                   memory-bandwidth-bound flattens after the first few.
  -lm mmap vs dio  load mode: mmap (warm) reads experts from the page cache
                   (RAM); dio bypasses the page cache, so every expert page-in
                   is a real NVMe read.  mmap - dio = the NVMe cost per token.
                   (Note: --moe-cache-mib cannot be used here: the fork rejects
                   it CPU-only with "MoE cache requires a GPU backend" [measured])

Signals per run, sampled from /proc while llama-bench runs:
  majflt           mmap'd model pages paged in (page-cache hits cost ~0)
  /proc/diskstats  actual NVMe sectors moved
  VmHWM            peak RSS

Reproduce one run:
    python3 bench/step1_attrib.py run --load-mode mmap --threads 8
Reproduce the whole attribution:
    python3 bench/step1_attrib.py sweep --out bench/results/step1-attrib.json
"""
import argparse
import json
import os
import subprocess
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LLAMA_BENCH = os.path.expanduser("~/dev/llama.cpp/build/bin/llama-bench")
DEFAULT_MODEL = os.path.join(REPO, "models", "DeepSeek-V2-Lite.Q4_K_M.gguf")
NVME = "nvme0n1"


def diskstats_sectors(dev=NVME):
    with open("/proc/diskstats") as fh:
        for line in fh:
            p = line.split()
            if len(p) > 10 and p[2] == dev:
                return int(p[5])  # sectors read
    return None


def sample_loop(pid, stop, out):
    """Poll /proc/<pid>/{io,stat,status} every 100 ms until the child exits."""
    out.update(rchar=0, read_bytes=0, majflt=0, minflt=0, vm_hwm_kib=0)
    while not stop.is_set():
        try:
            with open(f"/proc/{pid}/io") as fh:
                for line in fh:
                    if line.startswith("rchar:"):
                        out["rchar"] = max(out["rchar"], int(line.split()[1]))
                    elif line.startswith("read_bytes:"):
                        out["read_bytes"] = max(out["read_bytes"], int(line.split()[1]))
            with open(f"/proc/{pid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
                out["majflt"] = max(out["majflt"], int(fields[7]))
                out["minflt"] = max(out["minflt"], int(fields[9]))
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        out["vm_hwm_kib"] = max(out["vm_hwm_kib"], int(line.split()[1]))
        except (FileNotFoundError, ProcessLookupError):
            break
        time.sleep(0.1)


def run_once(bench, model, threads, load_mode, n_gen, reps, cache_k=None,
             cache_v=None, no_warmup=False):
    cmd = [bench, "-m", model, "-p", "0", "-n", str(n_gen), "-r", str(reps),
           "-t", str(threads), "-o", "json", "-lm", load_mode]
    if cache_k:
        cmd += ["-ctk", cache_k]
    if cache_v:
        cmd += ["-ctv", cache_v]
    if no_warmup:
        cmd.append("--no-warmup")

    sec0 = diskstats_sectors()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    stop = threading.Event()
    sample = {}
    th = threading.Thread(target=sample_loop, args=(proc.pid, stop, sample))
    th.start()
    out, _ = proc.communicate()
    stop.set()
    th.join()
    wall = time.time() - t0
    sec1 = diskstats_sectors()
    if proc.returncode != 0:
        raise SystemExit(f"llama-bench exited {proc.returncode} for {cmd}")

    rows = []
    for entry in json.loads(out):
        # no "test" field in the JSON: decode rows are the ones with n_gen > 0
        if entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
            rows.append({
                "threads": entry["n_threads"],
                "tok_s": round(entry["avg_ts"], 2),
                "std": round(entry.get("stddev_ts") or 0.0, 2),
                "per_rep_tok_s": [round(x, 2) for x in entry.get("samples_ts", [])],
            })
    return {
        "load_mode": load_mode, "n_gen": n_gen, "reps": reps,
        "cache_k": cache_k, "cache_v": cache_v, "no_warmup": no_warmup,
        "tg": rows,
        "majflt": sample.get("majflt", 0),
        "minflt": sample.get("minflt", 0),
        "nvme_mib_read": round((sec1 - sec0) * 512 / 2**20, 1) if None not in (sec0, sec1) else None,
        "read_bytes_mib": round(sample.get("read_bytes", 0) / 2**20, 1),
        "peak_rss_mib": round(sample.get("vm_hwm_kib", 0) / 1024, 0),
        "wall_s": round(wall, 1),
    }


def show(res, label=""):
    for row in res["tg"]:
        print(f"{label:<34} t{row['threads']:<2} {row['tok_s']:>7.2f} ± {row['std']:<5.2f} t/s  "
              f"majflt={res['majflt']:>7}  NVMe={res['nvme_mib_read'] or 0:>8.1f} MiB  "
              f"RSS={res['peak_rss_mib']:>6.0f} MiB  wall={res['wall_s']}s", flush=True)


def sweep(args):
    results = []

    def one(label, **kw):
        print(f"running: {label} ...", flush=True)
        r = run_once(args.llama_bench, args.model, n_gen=args.n_gen,
                     reps=args.reps, **kw)
        r["label"] = label
        show(r, label)
        results.append(r)
        return r

    q8 = dict(cache_k="q8_0", cache_v="q8_0")

    # (1) warm baseline at manifest cache types (matches the 12.20 -> 14.90 comparison)
    one("warm mmap q8_0 t8", threads=8, load_mode="mmap", **q8)
    # (2) the same with the default f16 KV cache, for reference
    one("warm mmap f16 t8", threads=8, load_mode="mmap")
    # (3) thread scaling: ALU-bound scales, bandwidth-bound flattens
    one("warm mmap q8_0 threads", threads="1,2,4", load_mode="mmap", **q8)
    # (4) dio: every expert page-in is a real NVMe read
    one("dio q8_0 t8", threads=8, load_mode="dio", **q8)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["run", "sweep"])
    ap.add_argument("--llama-bench", default=DEFAULT_LLAMA_BENCH)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", default="8")
    ap.add_argument("--load-mode", default="mmap",
                    choices=["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"])
    ap.add_argument("--n-gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(REPO, "bench/results/step1-attrib.json"))
    args = ap.parse_args()
    if args.mode == "run":
        res = run_once(args.llama_bench, args.model, args.threads,
                       args.load_mode, args.n_gen, args.reps,
                       cache_k="q8_0", cache_v="q8_0")
        show(res, f"{args.load_mode} q8_0")
    else:
        sweep(args)


if __name__ == "__main__":
    main()
