"""A small, bounded set of loaded engines for multi-model serving.

The gateway is single-threaded by design (one llama.cpp context, one request at
a time), so the pool does not need concurrency tricks. It does need two
guarantees the HTTP layer depends on:

* loading a model takes minutes, so it happens on the caller's thread and the
  async layer calls it through ``asyncio.to_thread`` — never on the event loop;
* an engine that is serving a request must not be closed to make room for
  another model, so eviction only ever picks an idle engine, and a load waits
  for a slot instead of taking one from a live request.

Loading is lazy — a model is planned and mounted on its first request, so
`--models a,b,c` does not pay for three models when clients only ever use one.
"""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from vipera.backends.protocol import ChatEngine


class UnknownModelError(KeyError):
    """The request named a model that is not part of this gateway."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(name)
        self.name = name
        self.available = available


class ModelPool:
    """Map alias -> engine, loading on demand and evicting least-recently-used.

    ``acquire()`` returns an engine and marks it in use; the caller must pair it
    with ``release()`` when the request is finished (including after a stream is
    closed). ``get()`` is the CLI convenience: acquire and release immediately,
    which still leaves the engine loaded until the next acquire needs its slot.
    """

    def __init__(
        self,
        paths: dict[str, str | Path],
        factory: Callable[[str, str | Path], ChatEngine],
        *,
        max_loaded: int = 2,
    ) -> None:
        if not paths:
            raise ValueError("a pool needs at least one model")
        self._paths = dict(paths)
        self._factory = factory
        self._max_loaded = max(1, int(max_loaded))
        self._loaded: OrderedDict[str, ChatEngine] = OrderedDict()
        # How many in-flight requests hold each engine. Eviction skips these.
        self._in_use: dict[str, int] = {}
        # Aliases being loaded in another thread, so two requests for the same
        # model do not both pay for the load.
        self._loading: set[str] = set()
        self._cond = threading.Condition(threading.Lock())

    @property
    def default(self) -> str:
        return next(iter(self._paths))

    def ids(self) -> list[str]:
        return list(self._paths)

    def loaded_ids(self) -> list[str]:
        with self._cond:
            return list(self._loaded)

    def resolve(self, name: str | None) -> str:
        """Map a request's `model` to an alias, or raise UnknownModelError."""
        alias = name or self.default
        if alias not in self._paths:
            raise UnknownModelError(alias, self.ids())
        return alias

    def acquire(self, name: str | None) -> tuple[ChatEngine, str]:
        """Return (engine, alias), loading and evicting idle engines as needed.

        This blocks while the model loads and while every resident engine is
        serving; callers on an event loop must run it in a worker thread.
        """
        alias = self.resolve(name)
        while True:
            evicted: ChatEngine | None = None
            with self._cond:
                engine = self._loaded.get(alias)
                if engine is not None:
                    self._loaded.move_to_end(alias)
                    self._in_use[alias] = self._in_use.get(alias, 0) + 1
                    return engine, alias
                if alias in self._loading:
                    self._cond.wait()
                    continue
                if len(self._loaded) + len(self._loading) >= self._max_loaded:
                    victim = next(
                        (name for name in self._loaded if not self._in_use.get(name)), None
                    )
                    if victim is None:
                        # Every resident model is serving a request. Closing one
                        # here would free a context llama.cpp is still using.
                        self._cond.wait()
                        continue
                    evicted = self._loaded.pop(victim)
                    self._in_use.pop(victim, None)
                self._loading.add(alias)
                self._cond.notify_all()

            # Close outside the lock: stopping a llama-server child can take
            # seconds, and other threads must still be able to release.
            if evicted is not None:
                with contextlib.suppress(Exception):
                    evicted.close()
            try:
                engine = self._factory(alias, self._paths[alias])
            except BaseException:
                with self._cond:
                    self._loading.discard(alias)
                    self._cond.notify_all()
                raise
            with self._cond:
                self._loading.discard(alias)
                self._loaded[alias] = engine
                self._in_use[alias] = 1
                self._cond.notify_all()
            return engine, alias

    def release(self, engine: ChatEngine) -> None:
        """Drop one in-flight reference; the engine may now be evicted."""
        with self._cond:
            for alias, loaded in self._loaded.items():
                if loaded is engine:
                    count = self._in_use.get(alias, 0)
                    if count <= 1:
                        self._in_use.pop(alias, None)
                    else:
                        self._in_use[alias] = count - 1
                    self._cond.notify_all()
                    return

    def get(self, name: str | None) -> tuple[ChatEngine, str]:
        """Return (engine, alias) for one-off callers such as the CLI."""
        engine, alias = self.acquire(name)
        self.release(engine)
        return engine, alias

    def close(self) -> None:
        with self._cond:
            engines = list(self._loaded.values())
            self._loaded.clear()
            self._in_use.clear()
            self._cond.notify_all()
        for engine in engines:
            with contextlib.suppress(Exception):
                engine.close()
