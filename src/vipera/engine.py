"""Compatibility import path for the backends package.

The implementation moved to `vipera.backends` (`protocol`, `loading`,
`registry`, `inprocess`, `server`). These names stay importable so callers and
scripts written against 1.x keep working; new code should import from
`vipera.backends.*` directly.
"""

from __future__ import annotations

from vipera.backends.inprocess import Engine
from vipera.backends.loading import (
    check_host_ram,
    layer_ladder,
    load_attempts,
    with_gpu_layers,
)
from vipera.backends.protocol import CHAT_PARAMS, DEFAULT_SAMPLING, ChatEngine
from vipera.backends.registry import create_engine, resolve_backend

# Private aliases kept for code that reached into the old module.
_CHAT_PARAMS = CHAT_PARAMS
_DEFAULT_SAMPLING = DEFAULT_SAMPLING

__all__ = [
    "CHAT_PARAMS",
    "DEFAULT_SAMPLING",
    "ChatEngine",
    "Engine",
    "check_host_ram",
    "create_engine",
    "layer_ladder",
    "load_attempts",
    "resolve_backend",
    "with_gpu_layers",
]
