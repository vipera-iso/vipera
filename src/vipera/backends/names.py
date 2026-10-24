"""Which backend names exist: the built-ins plus anything installed as a plugin.

Kept free of any engine import so `vipera.config` can validate a backend name
without pulling in llama.cpp or a subprocess.
"""

from __future__ import annotations

import importlib.metadata as metadata
from functools import lru_cache

GROUP = "vipera.backends"
# `auto` is not an engine: the registry resolves it to `server` when a CUDA
# llama-server can see the card, and to `inprocess` otherwise.
BUILTIN = ("auto", "inprocess", "server")


@lru_cache(maxsize=1)
def plugin_names() -> tuple[str, ...]:
    """Names registered through `[project.entry-points."vipera.backends"]`."""
    found = metadata.entry_points()
    # requires-python >= 3.11, so entry_points() always has .select.
    selected = found.select(group=GROUP)
    return tuple(sorted(ep.name for ep in selected if ep.name not in BUILTIN))


def names() -> tuple[str, ...]:
    return BUILTIN + plugin_names()


def is_known(name: str | None) -> bool:
    return bool(name) and name in names()
