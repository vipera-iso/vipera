"""What the machine has: CPU, RAM, and a GPU from any vendor.

The planner budgets layers against free VRAM, so every probe here reports free
memory whenever the vendor's own API will say. Probes run most-precise first:
NVML, amdgpu's sysfs counters, the `rocm-smi` / `xpu-smi` CLIs, and finally
`llama-server --list-devices` -- last because it starts a process, but the only
one that reports what the runtime Vipera will actually use can allocate, so it
covers cards that no userland tool knows about.

A probe that can only learn capacity scales it down and marks the number as an
estimate. Guessing is worse than saying so: the planner turns the guess into
layers, and `vipera system doctor` prints where the number came from.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

GB = 1024**3

# Where the kernel exposes amdgpu's memory counters on Linux.
_DRM_ROOT = Path("/sys/class/drm")

# Share of capacity a probe may assume is free when it cannot read free memory.
# A fifth held back covers the driver context and llama.cpp's compute buffers,
# which is roughly what the measured reserve came to on the reference 4 GB card
# (0.3 GB base + 7% of the card, see planner.py).
_UNMEASURED_FREE_FRACTION = 0.8

# An APU has a BIOS carve-out and a GTT window backed by system RAM; a discrete
# card reports its own VRAM and may still expose a GTT aperture. The smallest
# discrete VRAM in use is 1 GB, and an APU's default carve-out is below it, so
# only a sub-gigabyte carve-out counts as shared memory.
# shortcut: a BIOS set to a 1-2 GB UMA frame buffer is then budgeted as
# dedicated VRAM; that under-estimates, which costs layers instead of offloading
# onto a card that cannot hold them. Upgrade when a reliable APU flag exists.
_APU_CARVEOUT_CEILING_GB = 1.0

# `llama-server --list-devices` prints one line per card:
#   CUDA0: NVIDIA GeForce GTX 1650 (3730 MiB, 3600 MiB free)
# Older builds print only the capacity, with no free-memory reading:
#   CUDA0: NVIDIA GeForce GTX 1650 (3730 MiB)
_DEVICE_LINE = re.compile(
    r"^\s*(?P<backend>[A-Za-z]+)0:\s*(?P<name>.+?)\s*"
    r"\((?P<total>\d+)\s*MiB(?:,\s*(?P<free>\d+)\s*MiB free)?\)",
    re.MULTILINE,
)

# Words that identify the vendor when a device name or backend prefix carries
# it. A HIP build on AMD still calls its backend CUDA, so the name is checked
# before the prefix.
_VENDOR_WORDS = {
    "nvidia": "nvidia",
    "geforce": "nvidia",
    "rtx": "nvidia",
    "quadro": "nvidia",
    "tesla": "nvidia",
    "amd": "amd",
    "radeon": "amd",
    "instinct": "amd",
    "intel": "intel",
    "arc": "intel",
    "iris": "intel",
    "apple": "apple",
    "m1": "apple",
    "m2": "apple",
    "m3": "apple",
    "m4": "apple",
}

_BACKEND_VENDOR = {
    "cuda": "nvidia",
    "rocm": "amd",
    "hip": "amd",
    "sycl": "intel",
    "mtl": "apple",
    "metal": "apple",
}


@dataclass
class HardwareInfo:
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_free_gb: float
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    is_apple_silicon: bool
    # `nvidia` | `amd` | `intel` | `apple` | `none`
    gpu_vendor: str = "none"
    # True when the GPU addresses system RAM (Apple Silicon, AMD APU).
    unified_memory: bool = False
    # Which probe produced vram_free_gb: nvml, amdgpu-sysfs, rocm-smi,
    # xpu-smi, llama-server, unified, or none.
    vram_source: str = "none"
    # True when the probe could not read free memory and estimated it.
    vram_free_is_estimate: bool = False


@dataclass
class _Gpu:
    """One probe's answer. `total_gb == 0` means the probe saw nothing."""

    name: str
    vendor: str
    total_gb: float
    free_gb: float
    source: str
    unified: bool = False
    estimated: bool = False


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return text or None


def _run_json(cmd: list[str]) -> dict | None:
    try:
        probe = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        payload = json.loads(probe.stdout or "")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _walk(payload: object) -> Iterator[dict]:
    """Yield every dict in a vendor payload; their schemas move between versions."""
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _walk(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk(item)


def _number(record: dict, *needles: str, exclude: tuple[str, ...] = ()) -> float | None:
    """First numeric field whose key contains every needle and no excluded word.

    The exclusion matters for rocm-smi: "VRAM Total Memory (B)" and "VRAM Total
    Used Memory (B)" both contain "vram total", so key order decided whether the
    capacity or the usage won.
    """
    for key, value in record.items():
        lowered = key.lower()
        if all(needle in lowered for needle in needles) and not any(
            word in lowered for word in exclude
        ):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _text(record: dict, *needles: str) -> str | None:
    for key, value in record.items():
        lowered = key.lower()
        if (
            all(needle in lowered for needle in needles)
            and isinstance(value, str)
            and value.strip()
        ):
            return value.strip()
    return None


def _vendor_of(backend: str, name: str) -> str:
    lowered = name.lower()
    for word, vendor in _VENDOR_WORDS.items():
        if word in lowered:
            return vendor
    return _BACKEND_VENDOR.get(backend.lower(), "unknown")


def _from_nvml() -> _Gpu | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        finally:
            shutdown = getattr(pynvml, "nvmlShutdown", None)
            if shutdown is not None:
                shutdown()
    except Exception:  # an unusable NVML means "ask the next probe"
        return None
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    return _Gpu(
        name=name,
        vendor="nvidia",
        total_gb=memory.total / GB,
        free_gb=memory.free / GB,
        source="nvml",
    )


def _from_amdgpu_sysfs() -> _Gpu | None:
    for total_path in sorted(_DRM_ROOT.glob("card*/device/mem_info_vram_total")):
        device = total_path.parent
        vram_total = _read_int(total_path)
        if not vram_total:
            continue
        vram_used = _read_int(device / "mem_info_vram_used") or 0
        gtt_total = _read_int(device / "mem_info_gtt_total") or 0
        gtt_used = _read_int(device / "mem_info_gtt_used") or 0

        total_bytes = vram_total
        free_bytes = max(0, vram_total - vram_used)
        unified = gtt_total > vram_total and vram_total / GB < _APU_CARVEOUT_CEILING_GB
        if unified:
            total_bytes += gtt_total
            free_bytes += max(0, gtt_total - gtt_used)

        return _Gpu(
            name=_read_text(device / "product_name") or "AMD GPU",
            vendor="amd",
            total_gb=total_bytes / GB,
            free_gb=free_bytes / GB,
            source="amdgpu-sysfs",
            unified=unified,
        )
    return None


def _from_rocm_smi() -> _Gpu | None:
    payload = _run_json(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not payload:
        return None
    for card in _walk(payload):
        total = _number(card, "vram", "total", exclude=("used",))
        if not total:
            continue
        used = _number(card, "vram", "used") or 0.0
        return _Gpu(
            name=_text(card, "card", "series") or _text(card, "card", "model") or "AMD GPU",
            vendor="amd",
            total_gb=total / GB,
            free_gb=max(0.0, total - used) / GB,
            source="rocm-smi",
        )
    return None


def _from_xpu_smi() -> _Gpu | None:
    payload = _run_json(["xpu-smi", "discovery", "--json"])
    if not payload:
        return None
    for device in _walk(payload):
        total = _number(device, "memory", "physical")
        if not total:
            continue
        used = _number(device, "used", "memory")
        # Only trust a used-memory reading that fits inside the capacity: the
        # field's units have moved between xpu-smi releases.
        trustworthy_used = used if used and 0 < used <= total else None
        free = total - trustworthy_used if trustworthy_used else total * _UNMEASURED_FREE_FRACTION
        return _Gpu(
            name=_text(device, "device", "name") or "Intel GPU",
            vendor="intel",
            total_gb=total / GB,
            free_gb=free / GB,
            source="xpu-smi",
            estimated=trustworthy_used is None,
        )
    return None


def _find_server_binary() -> str | None:
    """A llama-server to ask, if one is installed. Imported lazily: the
    optimizer imports nothing from Vipera, but keeping this a seam lets tests
    answer the question without a binary on disk."""
    from vipera.optimizer import OptimizerError, find_llama_tool

    try:
        return find_llama_tool("server")
    except (OptimizerError, OSError):
        return None


def _from_llama_server() -> _Gpu | None:
    binary = _find_server_binary()
    if not binary:
        return None
    try:
        probe = subprocess.run(
            [binary, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _DEVICE_LINE.search(f"{probe.stdout}{probe.stderr}")
    if not match:
        return None
    total_mib = int(match.group("total"))
    free_mib = match.group("free")
    # A build that reports capacity only is still a usable GPU: budget a
    # conservative share of it and say the number is an estimate, instead of
    # falling through to CPU-only.
    usable_mib = int(free_mib) if free_mib is not None else total_mib * _UNMEASURED_FREE_FRACTION
    return _Gpu(
        name=match.group("name"),
        vendor=_vendor_of(match.group("backend"), match.group("name")),
        total_gb=total_mib / 1024,
        free_gb=usable_mib / 1024,
        source="llama-server",
        estimated=free_mib is None,
    )


# Tried in order. The first probe that sees a card wins, so a machine with an
# NVIDIA card and an Intel iGPU plans against the NVIDIA card.
_PROBES: tuple[Callable[[], _Gpu | None], ...] = (
    _from_nvml,
    _from_amdgpu_sysfs,
    _from_rocm_smi,
    _from_xpu_smi,
    _from_llama_server,
)

_NO_GPU = _Gpu(name="None", vendor="none", total_gb=0.0, free_gb=0.0, source="none")


def _first_gpu() -> _Gpu:
    for probe in _PROBES:
        gpu = probe()
        if gpu is not None and gpu.total_gb > 0:
            return gpu
    return _NO_GPU


def _apple_gpu(ram_total_gb: float, ram_free_gb: float) -> _Gpu:
    # Unified memory: the GPU can address most of RAM, but not all of it, and
    # Metal keeps headroom for the window server and the OS.
    return _Gpu(
        name="Apple Silicon (Unified Memory)",
        vendor="apple",
        total_gb=ram_total_gb,
        free_gb=ram_free_gb * 0.7,
        source="unified",
        unified=True,
    )


def detect() -> HardwareInfo:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "psutil is required for hardware detection. Install it with: pip install psutil"
        ) from exc

    cores = psutil.cpu_count(logical=False) or 1
    threads = psutil.cpu_count(logical=True) or cores
    vm = psutil.virtual_memory()
    ram_total = vm.total / GB
    ram_free = vm.available / GB

    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    gpu = _apple_gpu(ram_total, ram_free) if is_apple_silicon else _first_gpu()

    return HardwareInfo(
        cpu_cores=cores,
        cpu_threads=threads,
        ram_total_gb=ram_total,
        ram_free_gb=ram_free,
        gpu_name=gpu.name,
        vram_total_gb=gpu.total_gb,
        vram_free_gb=gpu.free_gb,
        is_apple_silicon=is_apple_silicon,
        gpu_vendor=gpu.vendor,
        unified_memory=gpu.unified,
        vram_source=gpu.source,
        vram_free_is_estimate=gpu.estimated,
    )
