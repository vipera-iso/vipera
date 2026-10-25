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
An idle model is closed after its ``keep_alive`` deadline, which mirrors
Ollama's daemon behaviour instead of keeping every model resident forever.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from vipera.backends.protocol import ChatEngine


class UnknownModelError(KeyError):
    """The request named a model that is not part of this gateway."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(name)
        self.name = name
        self.available = available


def engine_health(engine: ChatEngine) -> str:
    """One engine's own health word; a backend without a probe counts as alive."""
    probe = getattr(engine, "health", None)
    if probe is None:
        return "ok"
    try:
        return str(probe())
    except Exception:
        return "unreachable"


class ModelPool:
    """Map alias -> engine, loading on demand and evicting least-recently used.

    ``acquire()`` returns an engine and marks it in use; the caller must pair it
    with ``release()`` when the request is finished (including after a stream is
    closed). ``get()`` is the CLI convenience: acquire and release immediately,
    which still leaves the engine loaded until the next acquire needs its slot
    or its ``keep_alive`` deadline passes.
    """

    def __init__(
        self,
        paths: dict[str, str | Path],
        factory: Callable[[str, str | Path], ChatEngine],
        *,
        max_loaded: int = 2,
        keep_alive: float | None = 300.0,
        reap_interval: float = 1.0,
    ) -> None:
        if not paths:
            raise ValueError("a pool needs at least one model")
        self._paths = dict(paths)
        self._factory = factory
        self._max_loaded = max(1, int(max_loaded))
        self._keep_alive = keep_alive
        self._reap_interval = max(0.05, float(reap_interval))
        self._loaded: OrderedDict[str, ChatEngine] = OrderedDict()
        # How many in-flight requests hold each engine. Eviction skips these.
        self._in_use: dict[str, int] = {}
        # Aliases being loaded in another thread, so two requests for the same
        # model do not both pay for the load.
        self._loading: set[str] = set()
        # Monotonic deadlines and a per-request TTL waiting for release.
        self._expires_at: dict[str, float] = {}
        self._requested_ttl: dict[str, float | None] = {}
        self._cond = threading.Condition(threading.Lock())
        self._stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_loop, name="vipera-model-reaper", daemon=True
        )
        self._reaper.start()

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
                    # A request in flight owns the model; a stale deadline must
                    # not reap it while it is generating.
                    self._expires_at.pop(alias, None)
                    self._requested_ttl.pop(alias, None)
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
                    self._expires_at.pop(victim, None)
                    self._requested_ttl.pop(victim, None)
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
                self._expires_at.pop(alias, None)
                self._requested_ttl.pop(alias, None)
                self._cond.notify_all()
            return engine, alias

    def release(self, engine: ChatEngine) -> None:
        """Drop one in-flight reference; the engine may now be evicted."""
        expired: ChatEngine | None = None
        with self._cond:
            alias = self._alias_for_locked(engine)
            if alias is None:
                return
            count = self._in_use.get(alias, 0)
            if count > 1:
                self._in_use[alias] = count - 1
            else:
                self._in_use.pop(alias, None)
                ttl = self._requested_ttl.pop(alias, self._keep_alive)
                self._apply_expiry_locked(alias, ttl)
                deadline = self._expires_at.get(alias)
                if deadline is not None and deadline <= time.monotonic():
                    # keep_alive 0 means "free the model as soon as this request
                    # finishes", without waiting for the reaper tick.
                    expired = self._loaded.pop(alias, None)
                    self._expires_at.pop(alias, None)
            self._cond.notify_all()

        if expired is not None:
            with contextlib.suppress(Exception):
                expired.close()

    def keep_alive(self, engine: ChatEngine, seconds: float | None) -> None:
        """Set the TTL for ``engine``; ``None`` means the pool default.

        A request calls this before generation starts. While the engine is in
        use the TTL is parked in ``_requested_ttl`` and applied on release, so a
        zero TTL cannot close a context mid-generation.
        """
        with self._cond:
            alias = self._alias_for_locked(engine)
            if alias is None:
                return
            ttl = self._keep_alive if seconds is None else seconds
            if self._in_use.get(alias, 0) > 0:
                self._requested_ttl[alias] = ttl
            else:
                self._apply_expiry_locked(alias, ttl)

    def expire(self, now: float | None = None) -> list[str]:
        """Close idle engines whose deadline has passed. Returns their aliases.

        ``now`` is a ``time.monotonic()`` value; tests pass an explicit future
        value instead of sleeping.
        """
        current = time.monotonic() if now is None else now
        expired: list[tuple[str, ChatEngine]] = []
        with self._cond:
            for alias, deadline in list(self._expires_at.items()):
                if deadline > current or self._in_use.get(alias):
                    continue
                engine = self._loaded.pop(alias, None)
                self._expires_at.pop(alias, None)
                self._requested_ttl.pop(alias, None)
                if engine is not None:
                    expired.append((alias, engine))
            if expired:
                self._cond.notify_all()

        for _alias, engine in expired:
            with contextlib.suppress(Exception):
                engine.close()
        return [alias for alias, _engine in expired]

    def expiries(self) -> dict[str, str | None]:
        """Return loaded aliases mapped to an ISO deadline, or None for forever."""
        with self._cond:
            now_mono = time.monotonic()
            now_wall = datetime.now(UTC)
            result: dict[str, str | None] = {}
            for alias in self._loaded:
                deadline = self._expires_at.get(alias)
                if deadline is None:
                    result[alias] = None
                else:
                    remaining = max(0.0, deadline - now_mono)
                    stamp = (now_wall + timedelta(seconds=remaining)).isoformat()
                    result[alias] = stamp.replace("+00:00", "Z")
            return result

    def health(self) -> dict[str, str]:
        """Report each loaded engine's own health; never loads or restarts one.

        Only resident engines are probed: a health check that mounted a model
        would cause the load it is supposed to watch. Probing happens outside
        the lock because it is a network round trip, not bookkeeping.
        """
        with self._cond:
            loaded = list(self._loaded.items())
        return {alias: engine_health(engine) for alias, engine in loaded}

    def get(self, name: str | None) -> tuple[ChatEngine, str]:
        """Return (engine, alias) for one-off callers such as the CLI."""
        engine, alias = self.acquire(name)
        self.release(engine)
        return engine, alias

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            engines = list(self._loaded.values())
            self._loaded.clear()
            self._in_use.clear()
            self._expires_at.clear()
            self._requested_ttl.clear()
            self._cond.notify_all()
        for engine in engines:
            with contextlib.suppress(Exception):
                engine.close()
        if self._reaper.is_alive() and self._reaper is not threading.current_thread():
            self._reaper.join(timeout=5)

    def _alias_for_locked(self, engine: ChatEngine) -> str | None:
        for alias, loaded in self._loaded.items():
            if loaded is engine:
                return alias
        return None

    def _apply_expiry_locked(self, alias: str, ttl: float | None) -> None:
        if ttl is None or ttl < 0:
            self._expires_at.pop(alias, None)
        else:
            self._expires_at[alias] = time.monotonic() + ttl

    def _reap_loop(self) -> None:
        while not self._stop.wait(self._reap_interval):
            with contextlib.suppress(Exception):
                self.expire()
