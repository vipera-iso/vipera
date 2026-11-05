"""Which protocol layers exist and how one is mounted.

A protocol layer is a module exposing ``NAME`` and ``mount(app, gateway)``.
Built-ins are imported by name; anything else is looked up in the
``vipera.protocols`` entry-point group, so a third party can ship a layer
without touching this file:

    [project.entry-points."vipera.protocols"]
    anthropic = "my_pkg.anthropic"

``vipera.plugins`` stays import-light (config validates against it without
pulling FastAPI or a backend in).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from vipera.plugins import PROTOCOL_BUILTIN, load_protocol_plugin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from vipera.protocols.gateway import Gateway

_MODULES = {
    "openai": "vipera.protocols.openai",
    "ollama": "vipera.protocols.ollama",
}


def load(name: str):
    """Return the module (or plugin callable) implementing ``name``."""
    if name in _MODULES:
        return import_module(_MODULES[name])
    return load_protocol_plugin(name)


def mount(app: FastAPI, gateway: Gateway, selected=None) -> tuple[str, ...]:
    """Mount every selected protocol; returns the ones actually mounted."""
    requested = list(selected) if selected else list(PROTOCOL_BUILTIN[:1])
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
