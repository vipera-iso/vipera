"""Run a real model through llama.cpp using a vipera residency manifest.

The rest of vipera *decides* where each bucket of weights should live and writes
that decision to `*.vipera.json`. This module is the other half: it turns the
manifest into an actual llama.cpp command line, runs it, and measures what the
machine really did (throughput, peak RSS, disk read bytes).

The point is that vipera stops being a prediction tool here — `vipera run`
executes the model on the machine instead of modelling it.

llama.cpp is treated as an external binary and is never patched or rebuilt.
The mapping from manifest to flags is deliberately small:

    manifest field                    llama.cpp flag
    --------------------------------  --------------------------------------
    every MoE layer's experts in RAM   --n-cpu-moe <n_layers>
    expert cache in VRAM               --moe-cache-mib <expert_cache_vram_mb>
    attention / resident weights       -ngl <n_layers>
    kv_cache.type_k / type_v           -ctk / -ctv

There is no per-expert disk streaming in llama.cpp, so a plan with experts
spilled to disk is reported as unmappable rather than silently approximated.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

MIB = 1024 * 1024

# The env var bench/llama-moe-prefetch.patch reads to arm expert prefetching.
PREFETCH_ENV = "LLAMA_MOE_PREFETCH"

# llama.cpp libraries that mean a GPU backend is compiled into this build.
_GPU_BACKENDS = (
    "libggml-cuda.so", "libggml-vulkan.so", "libggml-hip.so",
    "libggml-metal.so", "libggml-sycl.so", "libggml-opencl.so",
    "ggml-cuda.dll", "ggml-vulkan.dll",
)

# llama-cli prints this on stdout at the end of a generation.
_THROUGHPUT_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]"
)

# bench/llama-moe-prefetch.patch's own verdict, on stderr at shutdown.
_PREFETCH_RE = re.compile(
    r"prefetch: placed = (\d+), useful = (\d+) \(([0-9.]+)%\), "
    r"extra upload = ([0-9.]+) MiB"
)


@dataclass
class RunFlags:
    """The llama.cpp arguments a manifest implies, plus what could not be mapped."""

    ngl: int | None = None
    n_cpu_moe: int | None = None
    moe_cache_mib: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    warnings: list[str] = field(default_factory=list)


def load_manifest(path: str) -> dict:
    """Read a `*.vipera.json` produced by `vipera plan` (or `schedule`)."""
    with open(path) as f:
        return json.load(f)


def detect_gpu_backend(binary: str) -> bool:
    """True when a GPU backend library sits next to the llama.cpp binary.

    The runtime prints "compiled without GPU support" at run time, but a runner
    needs the answer *before* choosing `-ngl` / `--moe-cache-mib`, so the build
    directory is inspected directly.
    """
    resolved = shutil.which(binary) or binary
    directory = os.path.dirname(os.path.abspath(resolved))
    try:
        present = os.listdir(directory)
    except OSError:
        return False
    lowered = {p.lower() for p in present}
    return any(backend in lowered for backend in _GPU_BACKENDS)


def _file_contains(path: str, needle: bytes) -> bool:
    """True when `needle` occurs in the file, read in bounded chunks."""
    try:
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                if needle in chunk:
                    return True
    except OSError:
        return False
    return False


def supports_prefetch(binary: str) -> bool:
    """True when the build carries the MoE prefetch patch.

    The patch reads `LLAMA_MOE_PREFETCH`, and the string only exists in a build
    that has it — so the runner can say "this binary will ignore the table"
    instead of silently prefetching nothing.

    The launcher is not where the code lives: with shared libraries `llama-cli`
    is a stub and the patch sits in `libllama.so` beside it, so the whole
    directory's libraries are searched, not just the executable.
    """
    resolved = shutil.which(binary) or binary
    needle = PREFETCH_ENV.encode()
    if _file_contains(resolved, needle):
        return True
    directory = os.path.dirname(os.path.abspath(resolved))
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    return any(
        ".so" in name and _file_contains(os.path.join(directory, name), needle)
        for name in entries
    )


def plan_flags(manifest: dict, *, gpu: bool, n_layers: int | None = None) -> RunFlags:
    """Translate a residency manifest into llama.cpp flags.

    `n_layers` is taken from the live model when known (more reliable than the
    manifest, which may have been written for a different file).
    """
    placement = manifest.get("placement") or {}
    source = manifest.get("source") or {}
    n = n_layers if n_layers is not None else source.get("n_layers")

    flags = RunFlags()
    warnings = flags.warnings

    cache_vram = int(placement.get("expert_cache_vram_bytes") or 0)
    resident_vram = int(placement.get("resident_vram_bytes") or 0)
    spill = int(placement.get("expert_spill_disk_bytes") or 0)

    if gpu:
        if n:
            # Offload every layer's attention, but keep every layer's experts on
            # the CPU — the model overflows VRAM, which is the case vipera targets.
            flags.ngl = n
            flags.n_cpu_moe = n
        if cache_vram > 0:
            flags.moe_cache_mib = cache_vram // MIB
    else:
        flags.ngl = 0
        if cache_vram > 0 or resident_vram > 0:
            warnings.append(
                "llama.cpp was built without a GPU backend, so the manifest's VRAM "
                f"placement ({resident_vram / MIB:.0f} MiB resident + "
                f"{cache_vram / MIB:.0f} MiB expert cache) cannot be honoured; "
                "running CPU-only with -ngl 0"
            )

    if spill > 0:
        warnings.append(
            f"{spill / MIB:.0f} MiB of experts are planned to spill to disk, and "
            "llama.cpp has no per-expert disk streaming — it will fall back to mmap "
            "paging, whose cost the manifest's modelled hit rate does not describe"
        )

    kv = manifest.get("kv_cache") or {}
    flags.cache_type_k = kv.get("type_k")
    flags.cache_type_v = kv.get("type_v")
    return flags


def build_command(
    binary: str,
    model: str,
    flags: RunFlags,
    *,
    context: int,
    n_predict: int,
    prompt: str,
    threads: int | None = None,
    extra: list[str] | None = None,
    prefetch: bool = False,
) -> list[str]:
    """The exact llama-cli command line for a manifest.

    `prefetch` adds `-v`: the MoE cache and the prefetch verdict only log at INFO
    level, which llama-cli hides by default, so a prefetch run that says nothing
    about its own hit rate is a run nobody can judge.
    """
    cmd = [
        binary, "-m", model,
        "-c", str(context),
        "-n", str(n_predict),
        "-p", prompt,
        # Single turn: finish after one generation instead of waiting on a TTY.
        "-st",
    ]
    if flags.ngl is not None:
        cmd += ["-ngl", str(flags.ngl)]
    if flags.n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(flags.n_cpu_moe)]
    if flags.moe_cache_mib is not None:
        cmd += ["--moe-cache-mib", str(flags.moe_cache_mib)]
    if flags.cache_type_k:
        cmd += ["-ctk", flags.cache_type_k]
    if flags.cache_type_v:
        cmd += ["-ctv", flags.cache_type_v]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if prefetch and "-v" not in (extra or []) and "--verbose" not in (extra or []):
        cmd.append("-v")
    cmd += extra or []
    return cmd


def parse_throughput(text: str) -> tuple[float | None, float | None]:
    """(prompt t/s, generation t/s) from llama-cli's stdout, or (None, None).

    Rebuilt from the last match in the stream: a warmup or an aborted turn can
    print the line more than once, and the final one describes the real run.
    """
    matches = _THROUGHPUT_RE.findall(text)
    if not matches:
        return None, None
    prompt_tps, gen_tps = matches[-1]
    return float(prompt_tps), float(gen_tps)


def parse_prefetch_stats(text: str) -> dict | None:
    """The patch's prefetch report, or None when prefetching did not run.

    `useful` is the share of prefetched experts the router went on to select, i.e.
    the share of the extra upload that bought something. A low value means the
    prediction table is wrong, not that the cache is.
    """
    m = _PREFETCH_RE.search(text)
    if not m:
        return None
    return {
        "placed": int(m.group(1)),
        "useful": int(m.group(2)),
        "useful_pct": float(m.group(3)),
        "extra_upload_mib": float(m.group(4)),
    }


class _Sampler:
    """Poll a running pid for peak RSS and cumulative disk read bytes.

    Only /proc is used, and every read is best-effort: a missing entry or a short
    lived process degrades to None rather than failing the run.
    """

    def __init__(self, pid: int, interval: float = 0.2):
        self.pid = pid
        self.interval = interval
        self.peak_rss_kb: int | None = None
        self.read_bytes: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _take(self) -> None:
        try:
            with open(f"/proc/{self.pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                        if self.peak_rss_kb is None or rss > self.peak_rss_kb:
                            self.peak_rss_kb = rss
                        break
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open(f"/proc/{self.pid}/io") as f:
                for line in f:
                    if line.startswith("read_bytes:"):
                        self.read_bytes = int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval)

    def start(self) -> "_Sampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._take()


def run(
    binary: str,
    model: str,
    flags: RunFlags,
    *,
    context: int,
    n_predict: int,
    prompt: str,
    threads: int | None = None,
    extra: list[str] | None = None,
    timeout: float | None = None,
    sample_interval: float = 0.2,
    extra_env: dict[str, str] | None = None,
) -> dict:
    """Run llama-cli for real and report throughput and resource cost.

    `extra_env` is merged over the current environment for the child only, which
    is how the prefetch patch is armed (`LLAMA_MOE_PREFETCH`).
    """
    cmd = build_command(
        binary, model, flags, context=context, n_predict=n_predict,
        prompt=prompt, threads=threads, extra=extra,
        prefetch=bool(extra_env and PREFETCH_ENV in extra_env),
    )

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        stdin=subprocess.DEVNULL, env=env,
    )
    sampler = _Sampler(proc.pid, interval=sample_interval).start()
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, err = proc.communicate()
    finally:
        sampler.stop()
    elapsed = time.monotonic() - start

    prompt_tps, gen_tps = parse_throughput(out or "")
    prefetch_stats = parse_prefetch_stats(err or "")
    report: dict = {
        "ok": proc.returncode == 0 and not timed_out,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 3),
        "command": cmd,
        "peak_rss_kb": sampler.peak_rss_kb,
        "disk_read_bytes": sampler.read_bytes,
        "prompt_tokens_per_s": prompt_tps,
        "generation_tokens_per_s": gen_tps,
        "n_predict": n_predict,
        "context": context,
        "env": dict(extra_env or {}),
    }
    if prefetch_stats:
        report["prefetch"] = prefetch_stats
    if timed_out:
        report["error"] = f"timed out after {timeout:.0f} s"
    elif proc.returncode is not None and proc.returncode < 0:
        # Negative means killed by a signal; -9 on this box is the OOM killer.
        signame = "SIGKILL" if proc.returncode == -9 else f"signal {-proc.returncode}"
        report["error"] = f"killed by {signame}"
    elif proc.returncode != 0:
        report["error"] = f"llama-cli exited {proc.returncode}"
    if report.get("error"):
        report["stderr_tail"] = (err or "")[-2000:]
    return report


def format_report(report: dict, *, flags: RunFlags, manifest_source: str) -> str:
    lines: list[str] = []
    lines.append(f"manifest      {manifest_source}")
    mapping = []
    if flags.ngl is not None:
        mapping.append(f"-ngl {flags.ngl}")
    if flags.n_cpu_moe is not None:
        mapping.append(f"--n-cpu-moe {flags.n_cpu_moe}")
    if flags.moe_cache_mib is not None:
        mapping.append(f"--moe-cache-mib {flags.moe_cache_mib}")
    if flags.cache_type_k or flags.cache_type_v:
        mapping.append(f"-ctk {flags.cache_type_k or '?'} -ctv {flags.cache_type_v or '?'}")
    lines.append(f"mapped to     {' '.join(mapping) if mapping else '(nothing)'}")
    for w in flags.warnings:
        lines.append(f"warning       {w}")
    lines.append(f"command       {' '.join(report['command'])}")
    lines.append("")
    lines.append(f"ok            {report['ok']}   (exit {report['exit_code']})")
    lines.append(f"wall          {report['elapsed_s']} s")
    if report.get("peak_rss_kb") is not None:
        lines.append(f"peak RSS      {report['peak_rss_kb'] / 1024:.1f} MiB")
    if report.get("disk_read_bytes") is not None:
        lines.append(f"disk read     {report['disk_read_bytes'] / MIB:.1f} MiB")
    pf = report.get("prefetch")
    if pf:
        lines.append(
            f"prefetch      {pf['useful']} of {pf['placed']} predicted experts used "
            f"({pf['useful_pct']:.1f}%), +{pf['extra_upload_mib']:.1f} MiB upload"
        )
    p = report.get("prompt_tokens_per_s")
    g = report.get("generation_tokens_per_s")
    lines.append(f"prompt        {p:.1f} t/s" if p is not None else "prompt        n/a")
    lines.append(f"generation    {g:.2f} t/s" if g is not None else "generation    n/a")
    if report.get("error"):
        lines.append(f"error         {report['error']}")
    return "\n".join(lines)


def compare(
    binary: str,
    model: str,
    flags: RunFlags,
    *,
    context: int,
    n_predict: int,
    prompt: str,
    threads: int | None = None,
    extra: list[str] | None = None,
    timeout: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    """Run the plain llama.cpp default and the manifest-driven command, then diff them.

    The baseline carries **no** manifest flags, so it is what llama.cpp does on its
    own — the comparison the project's "beats the default" claim has to be against.
    """
    common = dict(
        context=context, n_predict=n_predict, prompt=prompt,
        threads=threads, extra=extra, timeout=timeout,
    )
    baseline = run(binary, model, RunFlags(), **common)
    manifest = run(binary, model, flags, extra_env=extra_env, **common)
    return {
        "baseline": baseline,
        "manifest": manifest,
        "delta": _delta(baseline, manifest),
    }


def _delta(baseline: dict, manifest: dict) -> dict:
    b = baseline.get("generation_tokens_per_s")
    m = manifest.get("generation_tokens_per_s")
    out: dict = {}
    if b and m and b > 0:
        out["generation_speedup"] = round(m / b, 3)
    br, mr = baseline.get("peak_rss_kb"), manifest.get("peak_rss_kb")
    if br is not None and mr is not None:
        out["rss_delta_mib"] = round((mr - br) / 1024, 1)
    bd, md = baseline.get("disk_read_bytes"), manifest.get("disk_read_bytes")
    if bd is not None and md is not None:
        out["disk_read_delta_mib"] = round((md - bd) / MIB, 1)
    return out


def format_compare(report: dict, *, manifest_source: str,
                   prefetch: str | None = None) -> str:
    """Two-column diff of the plain default run and the manifest-driven one."""
    base = report["baseline"]
    man = report["manifest"]
    delta = report["delta"]

    def cell(value, fmt, unit=""):
        return f"{value:{fmt}}{unit}" if value is not None else "n/a"

    lines = [f"manifest      {manifest_source}"]
    if prefetch:
        lines.append(f"prefetch      {prefetch}")
    lines.append(f"baseline      {' '.join(base['command'])}")
    lines.append(f"manifest cmd  {' '.join(man['command'])}")
    lines.append("")
    header = f"{'metric':<16}{'baseline':>14}{'manifest':>14}{'delta':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    lines.append(f"{'generation t/s':<16}"
                 f"{cell(base.get('generation_tokens_per_s'), '.2f'):>14}"
                 f"{cell(man.get('generation_tokens_per_s'), '.2f'):>14}"
                 f"{cell(delta.get('generation_speedup'), '.2f') + ('x' if 'generation_speedup' in delta else ''):>14}")
    lines.append(f"{'prompt t/s':<16}"
                 f"{cell(base.get('prompt_tokens_per_s'), '.1f'):>14}"
                 f"{cell(man.get('prompt_tokens_per_s'), '.1f'):>14}"
                 f"{'':>14}")

    b_rss = base.get("peak_rss_kb")
    m_rss = man.get("peak_rss_kb")
    d_rss = delta.get("rss_delta_mib")
    lines.append(f"{'peak RSS':<16}"
                 f"{(f'{b_rss / 1024:.0f} MiB' if b_rss is not None else 'n/a'):>14}"
                 f"{(f'{m_rss / 1024:.0f} MiB' if m_rss is not None else 'n/a'):>14}"
                 f"{(f'{d_rss:+.0f} MiB' if d_rss is not None else ''):>14}")

    b_disk = base.get("disk_read_bytes")
    m_disk = man.get("disk_read_bytes")
    d_disk = delta.get("disk_read_delta_mib")
    lines.append(f"{'disk read':<16}"
                 f"{(f'{b_disk / MIB:.0f} MiB' if b_disk is not None else 'n/a'):>14}"
                 f"{(f'{m_disk / MIB:.0f} MiB' if m_disk is not None else 'n/a'):>14}"
                 f"{(f'{d_disk:+.0f} MiB' if d_disk is not None else ''):>14}")

    if "generation_speedup" in delta:
        lines.append("")
        lines.append(f"manifest / baseline generation speedup: {delta['generation_speedup']:.2f}x")
    for name, r in (("baseline", base), ("manifest", man)):
        if r.get("error"):
            lines.append(f"{name} error: {r['error']}")
    return "\n".join(lines)


def resolve_binary(binary: str) -> str:
    return shutil.which(binary) or binary


def default_binary() -> str:
    candidate = os.path.expanduser("~/dev/llama.cpp/build/bin/llama-cli")
    if os.path.exists(candidate):
        return candidate
    return "llama-cli"
