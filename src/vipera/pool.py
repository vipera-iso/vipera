"""A small, bounded set of loaded engines for multi-model serving.

The gateway is single-threaded by design (one llama.cpp context, one request at
a time), so the pool does not need concurrency tricks: callers already hold the
server lock while they ask for an engine. The lock here only guards `close()`
racing a load from a worker thread.

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
    """Map alias -> engine, loading on demand and evicting least-recently-used."""

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
        self._lock = threading.Lock()

    @property
    def default(self) -> str:
        return next(iter(self._paths))

    def ids(self) -> list[str]:
        return list(self._paths)

    def loaded_ids(self) -> list[str]:
        return list(self._loaded)

    def resolve(self, name: str | None) -> str:
        """Map a request's `model` to an alias, or raise UnknownModelError."""
        alias = name or self.default
        if alias not in self._paths:
            raise UnknownModelError(alias, self.ids())
        return alias

    def get(self, name: str | None) -> tuple[ChatEngine, str]:
        """Return (engine, alias), loading and evicting as needed."""
        alias = self.resolve(name)
        with self._lock:
            engine = self._loaded.pop(alias, None)
            if engine is None:
                engine = self._factory(alias, self._paths[alias])
            # Re-insert so OrderedDict keeps least-recently-used first.
            self._loaded[alias] = engine
            while len(self._loaded) > self._max_loaded:
                _, evicted = self._loaded.popitem(last=False)
                with contextlib.suppress(Exception):
                    evicted.close()
            return engine, alias

    def close(self) -> None:
        with self._lock:
            for engine in self._loaded.values():
                with contextlib.suppress(Exception):
                    engine.close()
            self._loaded.clear()
