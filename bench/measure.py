"""Sampling helpers for the benchmark harness.

Everything here is read-only and best-effort: a missing /proc entry or a missing
nvidia-smi must degrade to None, never crash a benchmark run.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field


def read_proc_io(pid: int) -> dict[str, int] | None:
    """Cumulative read/write byte counters for a pid, or None if unreadable."""
    try:
        with open(f"/proc/{pid}/io") as f:
            out: dict[str, int] = {}
            for line in f:
                key, _, value = line.partition(":")
                out[key.strip()] = int(value.strip())
            return out
    except (OSError, ValueError):
        return None


def read_rss_kb(pid: int) -> int | None:
    """Resident set size in KiB from /proc/<pid>/status, or None."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def read_gpu_used_mib(env: dict[str, str] | None = None) -> int | None:
    """Total used VRAM in MiB across GPUs, or None when there is no nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    total = 0
    seen = False
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += int(line)
            seen = True
        except ValueError:
            return None
    return total if seen else None


@dataclass
class Sample:
    rss_kb: int | None = None
    gpu_mib: int | None = None
    io: dict[str, int] | None = None


@dataclass
class Peak:
    rss_kb: int | None = None
    gpu_mib: int | None = None
    io: dict[str, int] | None = field(default=None)


class Sampler:
    """Poll a running pid in a background thread and keep the peaks.

    The last observed /proc/<pid>/io is kept as a lower bound on total disk I/O,
    because the counters vanish the moment the process exits.
    """

    def __init__(self, pid: int, interval: float = 0.2, gpu: bool = True):
        self.pid = pid
        self.interval = interval
        self.gpu = gpu
        self.peak = Peak()
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval)

    def _take(self) -> None:
        rss = read_rss_kb(self.pid)
        if rss is not None and (self.peak.rss_kb is None or rss > self.peak.rss_kb):
            self.peak.rss_kb = rss
        if self.gpu:
            gpu = read_gpu_used_mib()
            if gpu is not None and (self.peak.gpu_mib is None or gpu > self.peak.gpu_mib):
                self.peak.gpu_mib = gpu
        io = read_proc_io(self.pid)
        if io is not None:
            self.peak.io = io
        self.samples += 1

    def start(self) -> "Sampler":
        self._thread.start()
        return self

    def stop(self) -> Peak:
        self._stop.set()
        self._thread.join(timeout=2)
        self._take()  # one last look, in case the process is still alive
        return self.peak


def wait_for_exit(pid: int, poll: float = 0.05, timeout: float | None = None) -> bool:
    """Return True when the pid is gone, False on timeout."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if deadline is not None and time.monotonic() > deadline:
            return False
        time.sleep(poll)


def bytes_per_token(total_bytes: int | None, tokens: int) -> float | None:
    if total_bytes is None or tokens <= 0:
        return None
    return total_bytes / tokens
