"""What a protocol needs from the gateway, without importing ``vipera.server``.

A protocol layer only ever needs four things: which engine serves a request,
which aliases exist, which one is the default, and the lock that serialises
access to the model. Passing this object around keeps the route modules free of
the server module's internals (and keeps the dependency arrow one-way:
``server`` imports protocols, never the other way round).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Gateway:
    #: One llama.cpp context, one request at a time.
    lock: asyncio.Lock
    #: Resolve the `model` field of a request; raises UnknownModelError.
    engine_for: Callable[[str | None], Any]
    #: Every alias this process serves.
    ids: Callable[[], list[str]]
    #: The alias used when a request names none.
    default_model: Callable[[], str | None]
    #: Aliases that are loaded right now (pool mode) or the single model.
    loaded_ids: Callable[[], list[str]]
    #: The optimization layer (shaping + prefix accounting); None disables it.
    optimizer: Any = None
