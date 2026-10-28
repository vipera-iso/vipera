"""Choose the llama.cpp *launch* policy: load mode and thread count.

vipera already decides **where weights live** (the residency manifest). This
module decides **how the process loads them**, which is a different lever and,
on the target machine, a larger one. `plantodo.md` STEP 1 `[measured]`: the
default `mmap` path ran at **16.95 tok/s** while `mlock` and `dio` on the same
model and flags ran at **22.90** and **22.44** — because `mmap` re-faults pages
the OS had already evicted, and that paging is **26% of decode time** even
though the NVMe itself moves ≤0.2%.

Policy (size-driven; the reason is kept so a run can state its own condition):

* model fits RAM and the memlock limit covers it -> ``mlock`` (pin, no paging)
* model fits RAM but memlock is shorter          -> ``dio``   (bypass the page cache)
* model does not fit RAM                         -> ``mmap``  (paging is unavoidable)

Threads default to the **physical** core count: this workload is ALU-bound
(STEP 1 puts `mlock` at 43.7 ms/token against a 43.5 ms ALU floor), so the
hyperthreads add contention rather than throughput.

Every input is injectable, so the decision is testable without the real machine;
the thin ``read_*`` helpers fetch the live values, and ``choose_policy`` reads
them itself when a value is not passed in.
"""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass, field

# llama.cpp accepts these for `-lm` (verified in `llama-cli --help`,
# build 0.6.0-dev f2cc728): auto | mmap | mlock | mmap+mlock | dio.
LOAD_MODES = ("mmap", "mlock", "dio")

MIB = 1024 * 1024


@dataclass
class LaunchPolicy:
    threads: int
    load_mode: str
    reason: str
    warnings: list[str] = field(default_factory=list)

    def to_flags(self) -> list[str]:
        """The llama.cpp arguments this policy adds (`-t` is handled separately)."""
        return ["-lm", self.load_mode]


def read_available_ram_bytes(path: str = "/proc/meminfo") -> int | None:
    """MemAvailable in bytes, or None when it cannot be read.

    MemAvailable, not MemTotal: the machine has 15 GiB of RAM but only ~11.5 GiB
    is actually available, and the difference decides whether a model can be
    pinned at all.
    """
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024  # kB -> bytes
    return None


def read_memlock_limit_bytes() -> float:
    """RLIMIT_MEMLOCK soft limit in bytes; ``inf`` when unlimited."""
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (ValueError, OSError):
        return 0.0
    if soft == resource.RLIM_INFINITY:
        return float("inf")
    return float(soft)


def read_physical_cores(path: str = "/proc/cpuinfo") -> int | None:
    """Count distinct (physical id, core id) pairs, or None if unavailable.

    This separates 4 real cores from 8 logical threads on the target machine;
    `os.cpu_count()` reports the latter.
    """
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    seen: set[tuple[str, str]] = set()
    phys = core = None
    for line in text.splitlines():
        if not line.strip():
            if phys is not None and core is not None:
                seen.add((phys, core))
            phys = core = None
        elif line.startswith("physical id"):
            phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
    if phys is not None and core is not None:
        seen.add((phys, core))
    return len(seen) or None


def choose_policy(
    *,
    model_bytes: int,
    ram_bytes: int | None = None,
    memlock_bytes: float | None = None,
    cores: int | None = None,
    logical_cores: int | None = None,
    fit_margin: float = 0.95,
) -> LaunchPolicy:
    """Pick a load mode and thread count for `model_bytes` on this machine.

    `fit_margin` is the fraction of available RAM a model may occupy and still
    count as resident. Any argument left as None is read from the live machine.
    """
    if ram_bytes is None:
        ram_bytes = read_available_ram_bytes()
    if memlock_bytes is None:
        memlock_bytes = read_memlock_limit_bytes()
    if cores is None:
        cores = read_physical_cores()
    logical = logical_cores if logical_cores is not None else (os.cpu_count() or 1)

    # Threads: physical cores, capped by the logical count, at least one.
    physical = cores if cores else max(1, (logical + 1) // 2)
    threads = max(1, min(physical, logical))

    warnings: list[str] = []
    if ram_bytes is None:
        load_mode = "mmap"
        reason = (
            "available RAM could not be read, so the model cannot be shown to fit; "
            "staying on the default mmap path"
        )
        warnings.append("could not read MemAvailable; load mode left at mmap")
    elif model_bytes <= ram_bytes * fit_margin:
        if memlock_bytes >= model_bytes:
            load_mode = "mlock"
            reason = (
                f"model {model_bytes / MIB:.0f} MiB fits the {ram_bytes / MIB:.0f} MiB "
                "available RAM and the memlock limit allows pinning it, so no page "
                "can be evicted out from under decode"
            )
        else:
            load_mode = "dio"
            reason = (
                f"model {model_bytes / MIB:.0f} MiB fits the {ram_bytes / MIB:.0f} MiB "
                f"available RAM, but the memlock limit is only "
                f"{memlock_bytes / MIB:.0f} MiB so it cannot be pinned; DirectIO "
                "bypasses the page cache instead"
            )
            warnings.append(
                f"memlock limit {memlock_bytes / MIB:.0f} MiB < model "
                f"{model_bytes / MIB:.0f} MiB; using dio rather than mlock"
            )
    else:
        load_mode = "mmap"
        reason = (
            f"model {model_bytes / MIB:.0f} MiB does not fit the "
            f"{ram_bytes / MIB:.0f} MiB available RAM (>{fit_margin:.0%}), so paging "
            "is unavoidable; mmap lets the OS manage it"
        )
        warnings.append(
            f"model {model_bytes / MIB:.0f} MiB exceeds available RAM "
            f"{ram_bytes / MIB:.0f} MiB; expected to page"
        )

    return LaunchPolicy(threads=threads, load_mode=load_mode, reason=reason, warnings=warnings)


def describe(policy: LaunchPolicy) -> str:
    """One line for a report, so a run states why it loaded the way it did."""
    return f"launch policy  -lm {policy.load_mode} -t {policy.threads}  ({policy.reason})"
