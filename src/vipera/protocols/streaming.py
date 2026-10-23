"""Run a synchronous engine without blocking the event loop, under the lock.

Both HTTP shapes have the same two requirements:

* a single llama.cpp context means one generation at a time, so the gateway
  lock is held for the whole call;
* client disconnects cancel the request while the worker thread may still be
  inside ``next()`` — the lock must not be handed to the next request until
  that thread returns.

Keeping that logic here (instead of once per protocol) means a translation
layer cannot accidentally relax it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

_SENTINEL = object()


def _close_iterator(iterator: Iterator[Any]) -> None:
    """Close the iterator when it is a generator; plain iterators have no close()."""
    close = getattr(iterator, "close", None)
    if close is not None:
        with contextlib.suppress(Exception):
            close()


async def run_locked(lock: asyncio.Lock, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a blocking engine function in a worker thread while holding ``lock``."""
    await lock.acquire()
    released = False
    try:
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            # shield: cancelling the request must not cancel the worker thread,
            # because the engine may be mid-generation.
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(lambda _t: lock.release())
            released = True
            raise
    finally:
        if not released:
            lock.release()


async def iter_chunks(
    lock: asyncio.Lock,
    factory: Callable[[], Iterator[Any]],
) -> AsyncIterator[Any]:
    """Yield every item of a sync generator, pulling it in a worker thread.

    ``factory`` is called while the lock is held; for an engine that returns a
    generator this does no work until the first ``next()``.
    """
    await lock.acquire()
    released = False
    iterator = None
    try:
        iterator = factory()
        while True:
            task = asyncio.create_task(asyncio.to_thread(next, iterator, _SENTINEL))
            try:
                item = await asyncio.shield(task)
            except asyncio.CancelledError:
                # The worker is still inside next(); closing the generator now
                # would raise "already executing" and release the lock too
                # early. Wait for the thread, then close and release.
                def _finish(_t: Any, it: Iterator[Any] = iterator) -> None:
                    _close_iterator(it)
                    lock.release()

                task.add_done_callback(_finish)
                released = True
                iterator = None
                raise
            if item is _SENTINEL:
                break
            yield item
    finally:
        if iterator is not None:
            _close_iterator(iterator)
        if not released:
            lock.release()
