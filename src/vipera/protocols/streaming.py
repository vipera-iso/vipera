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
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

_SENTINEL = object()


class Lease:
    """One acquired engine, released exactly once when its request ends.

    ``start()`` is called after the gateway lock is held and just before the
    engine call reaches a worker thread or a stream iterator. If a request is
    cancelled before that (waiting for the lock), the lease counts as unused and
    may be released at once. Once started, only the wrapper around the engine
    call releases it — after that call returns or its iterator is closed — which
    is what keeps an eviction from freeing a context mid-generation.
    """

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._started = False
        self._released = False
        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release()

    def release_if_unstarted(self) -> None:
        """Backstop for a request that never reached the engine."""
        if not self._started:
            self.release()


def leased_call(lease: Lease, fn: Callable[..., Any], *args: Any, **kwargs: Any):
    """Wrap a blocking engine call so it releases the lease when it returns."""

    def _call():
        try:
            return fn(*args, **kwargs)
        finally:
            lease.release()

    return _call


def leased_iterator(lease: Lease, factory: Callable[[], Iterator[Any]]):
    """Wrap a sync iterator so closing it (or exhausting it) releases the lease."""

    def _factory():
        def _iterate():
            try:
                yield from factory()
            finally:
                lease.release()

        return _iterate()

    return _factory


def _close_iterator(iterator: Iterator[Any]) -> None:
    """Close the iterator when it is a generator; plain iterators have no close()."""
    close = getattr(iterator, "close", None)
    if close is not None:
        with contextlib.suppress(Exception):
            close()


async def run_locked(
    lock: asyncio.Lock,
    fn: Callable[..., Any],
    *args: Any,
    on_start: Callable[[], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Call a blocking engine function in a worker thread while holding ``lock``."""
    await lock.acquire()
    released = False
    try:
        if on_start is not None:
            on_start()
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
    *,
    on_start: Callable[[], None] | None = None,
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
        if on_start is not None:
            on_start()
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


async def run_leased(
    lock: asyncio.Lock,
    lease: Lease,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """``run_locked`` plus a lease: start it under the lock, release afterwards."""
    try:
        return await run_locked(lock, leased_call(lease, fn, *args, **kwargs), on_start=lease.start)
    except BaseException:
        # The lock was never acquired, so no worker can be using the engine.
        lease.release_if_unstarted()
        raise
