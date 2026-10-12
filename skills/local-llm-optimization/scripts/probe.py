#!/usr/bin/env python3
"""Probe local hardware and recommend a starting quantization / placement.

Best-effort and read-only: a missing /proc entry or nvidia-smi degrades to
"unknown" instead of crashing. Pure stdlib so it runs anywhere.

    python3 probe.py            # human table
    python3 probe.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

MIB = 1024 * 1024


def cpu_info() -> dict:
    info = {"model": None, "cores": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and info["model"] is None:
                    info["model"] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return info


def mem_info() -> dict:
    total_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    return {"total_mib": (total_kb // 1024) if total_kb else None}


def gpu_info() -> list[dict]:
    """NVIDIA only, via nvidia-smi. Empty list when absent."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []
    gpus = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append({"name": parts[0], "vram_mib": int(parts[1]),
                             "vram_free_mib": int(parts[2])})
            except ValueError:
                continue
    return gpus


def recommend(mem_mib: int | None, vram_mib: int | None) -> dict:
    """A conservative starting point, not a promise.

    Budget rule: fit the weights in VRAM if possible, else VRAM+RAM, and keep
    ~2 GiB headroom for KV/activations/runtime.
    """
    ram_gib = (mem_mib / 1024) if mem_mib else None
    vram_gib = (vram_mib / 1024) if vram_mib else 0
    usable = (ram_gib - 2) + max(0.0, vram_gib - 0.5) if ram_gib else None
    if usable is None:
        return {"note": "RAM unknown; cannot recommend"}
    # rough bytes/param: Q4_K_M ~0.55 GB per 1B params incl. overhead
    max_params_q4 = max(0.0, usable) / 0.55
    max_params_q8 = max(0.0, usable) / 1.05
    return {
        "usable_gib": round(usable, 1),
        "approx_max_params_b_q4_k_m": round(max_params_q4, 1),
        "approx_max_params_b_q8_0": round(max_params_q8, 1),
        "starting_quant": "Q4_K_M",
        "note": "Q4_K_M is the quality/size default; drop to Q3_K_M only if Q4 does not fit",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cpu, mem, gpus = cpu_info(), mem_info(), gpu_info()
    vram = max((g["vram_mib"] for g in gpus), default=None)
    rec = recommend(mem["total_mib"], vram)

    report = {"cpu": cpu, "memory": mem, "gpus": gpus, "recommendation": rec}
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print("hardware probe")
    print(f"  CPU        {cpu['model'] or 'unknown'}  ({cpu['cores']} threads)")
    print(f"  RAM        {('%d MiB' % mem['total_mib']) if mem['total_mib'] else 'unknown'}")
    if gpus:
        for g in gpus:
            print(f"  GPU        {g['name']}  {g['vram_mib']} MiB "
                  f"({g['vram_free_mib']} MiB free)")
    else:
        print("  GPU        none detected (CPU-only or non-NVIDIA)")
    print()
    print("starting recommendation (heuristic, not a promise)")
    for k, v in rec.items():
        print(f"  {k:<28} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
