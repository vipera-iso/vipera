"""Which protocol names exist: the built-ins plus anything installed as a plugin.

Kept free of FastAPI and engine imports so ``vipera.config`` can validate a
``protocols`` value without importing the web stack.
"""

from __future__ import annotations

import importlib.metadata as metadata
from functools import lru_cache

GROUP = "vipera.protocols"
BUILTIN = ("openai", "ollama")


@lru_cache(maxsize=1)
def plugin_names() -> tuple[str, ...]:
    """Names registered through ``[project.entry-points."vipera.protocols"]``."""
    found = metadata.entry_points()
    # requires-python >= 3.11, so entry_points() always has .select.
    selected = found.select(group=GROUP)
    return tuple(sorted(ep.name for ep in selected if ep.name not in BUILTIN))


def names() -> tuple[str, ...]:
    return BUILTIN + plugin_names()


def is_known(name: str | None) -> bool:
    return bool(name) and name in names()
