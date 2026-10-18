import contextlib
import platform
from dataclasses import dataclass


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


def detect() -> HardwareInfo:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "psutil is required for hardware detection. "
            "Install it with: pip install psutil"
        ) from exc

    cores = psutil.cpu_count(logical=False) or 1
    threads = psutil.cpu_count(logical=True) or cores
    vm = psutil.virtual_memory()
    ram_total = vm.total / 1024**3
    ram_free = vm.available / 1024**3

    is_mac = platform.system() == "Darwin"
    is_apple_silicon = is_mac and platform.machine() == "arm64"

    gpu_name, vram_total, vram_free = "None", 0.0, 0.0

    if is_apple_silicon:
        gpu_name = "Apple Silicon (Unified Memory)"
        vram_total = ram_total
        vram_free = ram_free * 0.7
    else:
        with contextlib.suppress(Exception):
            import pynvml

            pynvml.nvmlInit()
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(h)
                gpu_name = name.decode() if isinstance(name, bytes) else name
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                vram_total = mem.total / 1024**3
                vram_free = mem.free / 1024**3
            finally:
                pynvml.nvmlShutdown()

    return HardwareInfo(
        cpu_cores=cores, cpu_threads=threads,
        ram_total_gb=ram_total, ram_free_gb=ram_free,
        gpu_name=gpu_name,
        vram_total_gb=vram_total, vram_free_gb=vram_free,
        is_apple_silicon=is_apple_silicon,
    )
