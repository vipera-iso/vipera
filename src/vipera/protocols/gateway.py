"""What a protocol needs from the gateway, without importing ``vipera.server``.

A protocol layer only ever needs four things: which engine serves a request,
which aliases exist, which one is the default, and the lock that serialises
access to the model. Passing this object around keeps the route modules free of
the server module's internals (and keeps the dependency arrow one-way:
``server`` imports protocols, never the other way round).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


def _noop_release(engine: Any) -> None:
    """Single-model gateways never evict, so releasing is a no-op."""


#: One permit per concurrent request. A Lock is a Semaphore with one slot; the
#: server backend hands out `parallel` permits so llama.cpp slots can batch.
Permit = asyncio.Lock | asyncio.Semaphore


@dataclass(frozen=True)
class Gateway:
    #: One permit per concurrent request (one llama.cpp context, unless the
    #: backend runs several slots).
    lock: Permit
    #: Resolve the `model` field of a request; raises UnknownModelError. In pool
    #: mode this loads in a worker thread, so it is always awaited.
    engine_for: Callable[[str | None], Awaitable[Any]]
    #: Every alias this process serves.
    ids: Callable[[], list[str]]
    #: The alias used when a request names none.
    default_model: Callable[[], str | None]
    #: Aliases that are loaded right now (pool mode) or the single model.
    loaded_ids: Callable[[], list[str]]
    #: The optimization layer (shaping + prefix accounting); None disables it.
    optimizer: Any = None
    #: Give back one in-flight reference after a request (or its stream) ends.
    release: Callable[[Any], None] = _noop_release
