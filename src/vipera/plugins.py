"""Which backend and protocol names exist: the built-ins plus installed plugins.

Both subsystems answer the same question -- "is this name real, and what does it
point at?" -- so the entry-point lookup lives here once. Import-light on
purpose (stdlib only): `vipera.config` validates a name before anything heavy
(``llama.cpp``, FastAPI, a subprocess) is imported.
"""

from __future__ import annotations

import importlib.metadata as metadata
from functools import cache

BACKEND_GROUP = "vipera.backends"
# `auto` is not an engine: the registry resolves it to `server` when a CUDA
# llama-server can see the card, and to `inprocess` otherwise.
BACKEND_BUILTIN = ("auto", "inprocess", "server")

PROTOCOL_GROUP = "vipera.protocols"
PROTOCOL_BUILTIN = ("openai", "ollama")


@cache
def installed(group: str, builtin: tuple[str, ...]) -> tuple[str, ...]:
    """Entry-point names in `group` that `builtin` does not already cover.

    Cached per group: `entry_points()` walks every installed distribution, and
    the answer cannot change inside one process.
    """
    selected = metadata.entry_points().select(group=group)
    return tuple(sorted(ep.name for ep in selected if ep.name not in builtin))


def backend_names() -> tuple[str, ...]:
    return BACKEND_BUILTIN + installed(BACKEND_GROUP, BACKEND_BUILTIN)


def protocol_names() -> tuple[str, ...]:
    return PROTOCOL_BUILTIN + installed(PROTOCOL_GROUP, PROTOCOL_BUILTIN)


def is_known_backend(name: str | None) -> bool:
    return bool(name) and name in backend_names()


def is_known_protocol(name: str | None) -> bool:
    return bool(name) and name in protocol_names()


def load_protocol_plugin(name: str):
    """The module a third party registered for protocol `name`."""
    selected = metadata.entry_points().select(group=PROTOCOL_GROUP)
    for entry in selected:
        if entry.name == name:
            return entry.load()
    raise ValueError(f"unknown protocol '{name}' (known: {', '.join(protocol_names())})")
