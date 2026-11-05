"""NVML reads through `ctypes`, without importing `pynvml`.

`pynvml` is itself a thin `ctypes` wrapper, and importing it costs about 20 ms on
this machine -- more than the NVML calls it wraps. Vipera needs three of them, so
a GPU probe pays for the driver library and nothing else.

Every path returns `None` when NVML is absent or answers with an error, so the
caller asks the next probe instead of failing.
"""

from __future__ import annotations

import ctypes

# Driver library names, most specific first: the Linux soname the package ships,
# the Linux development symlink, then the Windows DLL.
_LIBRARIES = ("libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll")

# `nvmlDeviceGetName` wants a C string buffer; NVML caps a device name at 96 bytes.
_NAME_BYTES = 96

_SUCCESS = 0


class Memory(ctypes.Structure):
    """`nvmlMemory_t`: total, free and used device memory, in bytes."""

    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load() -> ctypes.CDLL | None:
    for name in _LIBRARIES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def probe(index: int = 0) -> tuple[str, int, int] | None:
    """Name, total and free bytes of one device, or `None` if NVML will not say."""
    library = _load()
    if library is None:
        return None

    # The `_v2` entry points take an index; the plain ones are the older ABI and
    # are still what a pre-2019 driver exports.
    init = getattr(library, "nvmlInit_v2", None) or getattr(library, "nvmlInit", None)
    handle_of = getattr(library, "nvmlDeviceGetHandleByIndex_v2", None) or getattr(
        library, "nvmlDeviceGetHandleByIndex", None
    )
    get_name = getattr(library, "nvmlDeviceGetName", None)
    get_memory = getattr(library, "nvmlDeviceGetMemoryInfo", None)
    shutdown = getattr(library, "nvmlShutdown", None)
    if init is None or handle_of is None or get_name is None or get_memory is None:
        return None

    for function in (init, handle_of, get_name, get_memory):
        function.restype = ctypes.c_int
    handle_of.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    get_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory)]

    if init() != _SUCCESS:
        return None
    try:
        handle = ctypes.c_void_p()
        if handle_of(index, ctypes.byref(handle)) != _SUCCESS:
            return None
        buffer = ctypes.create_string_buffer(_NAME_BYTES)
        if get_name(handle, buffer, _NAME_BYTES) != _SUCCESS:
            return None
        memory = Memory()
        if get_memory(handle, ctypes.byref(memory)) != _SUCCESS:
            return None
    finally:
        if shutdown is not None:
            shutdown.restype = ctypes.c_int
            shutdown()

    return buffer.value.decode(errors="replace"), memory.total, memory.free
