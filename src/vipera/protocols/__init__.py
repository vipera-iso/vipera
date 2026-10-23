"""Which protocol layers exist and how one is mounted.

A protocol layer is a module exposing ``NAME`` and ``mount(app, gateway)``.
Built-ins are imported by name; anything else is looked up in the
``vipera.protocols`` entry-point group, so a third party can ship a layer
without touching this file:

    [project.entry-points."vipera.protocols"]
    anthropic = "my_pkg.anthropic"

``vipera.protocols.names`` stays import-light (config validates against it
without pulling FastAPI or a backend in).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from vipera.protocols.names import BUILTIN, GROUP, names

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from vipera.protocols.gateway import Gateway

_MODULES = {
    "openai": "vipera.protocols.openai",
    "ollama": "vipera.protocols.ollama",
}


def _plugin_mount(name: str):
    from importlib.metadata import entry_points

    found = entry_points()
    # requires-python >= 3.11, so entry_points() always has .select.
    selected = found.select(group=GROUP)
    for entry in selected:
        if entry.name == name:
            return entry.load()
    raise ValueError(f"unknown protocol '{name}' (known: {', '.join(names())})")


def load(name: str):
    """Return the module (or plugin callable) implementing ``name``."""
    if name in _MODULES:
        return import_module(_MODULES[name])
    return _plugin_mount(name)


def mount(app: FastAPI, gateway: Gateway, selected=None) -> tuple[str, ...]:
    """Mount every selected protocol; returns the ones actually mounted."""
    requested = list(selected) if selected else list(BUILTIN[:1])
    mounted: list[str] = []
    for name in requested:
        name = str(name).strip().lower()
        if not name or name in mounted:
            continue
        protocol = load(name)
        mount_fn = getattr(protocol, "mount", protocol)
        mount_fn(app, gateway)
        mounted.append(name)
    return tuple(mounted)
