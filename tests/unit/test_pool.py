import threading
import time
from datetime import UTC, datetime

import pytest

from vipera.pool import ModelPool, ModelWaitTimeout, UnknownModelError


class FakeEngine:
    def __init__(self, alias):
        self.alias = alias
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def loads():
    return []


@pytest.fixture
def pool(loads):
    def factory(alias, path):
        loads.append(alias)
        return FakeEngine(alias)

    return ModelPool({"a": "/tmp/a.gguf", "b": "/tmp/b.gguf"}, factory, max_loaded=2)


def test_acquire_gives_up_when_a_load_never_finishes():
    # A wedged load used to park every later request for the life of the process:
    # the wait had no deadline at all.
    entered = threading.Event()
    release = threading.Event()

    def slow_factory(alias, path):
        entered.set()
        release.wait(10)
        return FakeEngine(alias)

    pool = ModelPool({"a": "/a"}, slow_factory, acquire_timeout=0.05)
    loader = threading.Thread(target=pool.acquire, args=("a",), daemon=True)
    loader.start()
    try:
        assert entered.wait(5)
        with pytest.raises(ModelWaitTimeout):
            pool.acquire("a")
    finally:
        release.set()
        loader.join(5)
        pool.close()


def test_ids_lists_every_configured_model(pool):
    assert pool.ids() == ["a", "b"]


def test_nothing_is_loaded_before_the_first_request(pool, loads):
    assert loads == []
    assert pool.loaded_ids() == []


def test_get_loads_once_and_reuses_the_engine(pool, loads):
    first, alias = pool.get("b")
    again, _ = pool.get("b")

    assert alias == "b"
    assert first is again
    assert loads == ["b"]


def test_missing_model_name_falls_back_to_the_first_alias(pool):
    engine, alias = pool.get(None)

    assert alias == "a"
    assert engine.alias == "a"


def test_unknown_model_reports_what_is_available(pool):
    with pytest.raises(UnknownModelError) as exc:
        pool.get("nope")

    assert exc.value.name == "nope"
    assert exc.value.available == ["a", "b"]


def test_least_recently_used_engine_is_evicted_and_closed():
    loaded = []

    def factory(alias, path):
        engine = FakeEngine(alias)
        loaded.append(engine)
        return engine

    pool = ModelPool({"a": "/a", "b": "/b", "c": "/c"}, factory, max_loaded=2)

    first, _ = pool.get("a")
    pool.get("b")
    pool.get("c")  # pushes "a" out

    assert first.closed is True
    assert pool.loaded_ids() == ["b", "c"]


def test_recently_used_engine_survives_eviction():
    pool = ModelPool(
        {"a": "/a", "b": "/b", "c": "/c"}, lambda alias, path: FakeEngine(alias), max_loaded=2
    )

    a, _ = pool.get("a")
    pool.get("b")
    pool.get("a")  # touch "a" so "b" is now the oldest
    pool.get("c")

    assert a.closed is False
    assert pool.loaded_ids() == ["a", "c"]


def test_an_engine_serving_a_request_is_never_evicted():
    """Eviction used to close the LRU engine even while a request used it.

    With max_loaded=1, loading "b" while "a" is in flight must wait for the
    release instead of closing a llama.cpp context mid-generation.
    """
    pool = ModelPool({"a": "/a", "b": "/b"}, lambda alias, path: FakeEngine(alias), max_loaded=1)
    in_flight, _ = pool.acquire("a")

    loaded = threading.Thread(target=pool.acquire, args=("b",), daemon=True)
    loaded.start()
    loaded.join(timeout=0.2)

    assert loaded.is_alive(), "the load must wait for the in-flight engine"
    assert in_flight.closed is False
    assert pool.loaded_ids() == ["a"]

    pool.release(in_flight)
    loaded.join(timeout=5)

    assert not loaded.is_alive()
    assert in_flight.closed is True
    assert pool.loaded_ids() == ["b"]


def test_reusing_an_in_flight_engine_does_not_evict_it():
    pool = ModelPool(
        {"a": "/a", "b": "/b", "c": "/c"},
        lambda alias, path: FakeEngine(alias),
        max_loaded=2,
    )
    first, _ = pool.acquire("a")
    second, _ = pool.acquire("a")

    assert first is second
    assert first.closed is False

    # Two acquires, two releases: the reference count must return to zero, or
    # the engine would look busy forever and no later load could evict it.
    pool.release(first)
    pool.release(second)
    pool.get("b")
    pool.get("c")

    assert first.closed is True


def test_two_requests_for_the_same_model_load_it_once():
    loads = []
    started = threading.Barrier(2)

    def factory(alias, path):
        loads.append(alias)
        return FakeEngine(alias)

    pool = ModelPool({"a": "/a"}, factory, max_loaded=1)
    engines = []

    def request():
        started.wait(timeout=5)
        engines.append(pool.acquire("a")[0])

    threads = [threading.Thread(target=request, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert loads == ["a"]
    assert len(engines) == 2
    assert engines[0] is engines[1]


def test_close_closes_every_loaded_engine(pool):
    a, _ = pool.get("a")
    b, _ = pool.get("b")

    pool.close()

    assert a.closed and b.closed
    assert pool.loaded_ids() == []


def test_idle_engine_is_closed_after_keep_alive():
    pool = ModelPool(
        {"a": "/a"},
        lambda alias, path: FakeEngine(alias),
        max_loaded=1,
        keep_alive=30,
        reap_interval=60,
    )
    engine, _ = pool.acquire("a")
    pool.release(engine)

    assert pool.expire(now=time.monotonic() + 31) == ["a"]
    assert engine.closed is True
    assert pool.loaded_ids() == []


def test_engine_in_use_is_never_reaped():
    pool = ModelPool(
        {"a": "/a"},
        lambda alias, path: FakeEngine(alias),
        max_loaded=1,
        keep_alive=30,
        reap_interval=60,
    )
    engine, _ = pool.acquire("a")
    pool.keep_alive(engine, 0)

    assert pool.expire(now=time.monotonic() + 10_000) == []
    assert engine.closed is False

    pool.release(engine)

    assert engine.closed is True


def test_keep_alive_zero_unloads_when_the_request_finishes():
    pool = ModelPool(
        {"a": "/a"},
        lambda alias, path: FakeEngine(alias),
        max_loaded=1,
        keep_alive=30,
        reap_interval=60,
    )
    engine, _ = pool.acquire("a")
    pool.keep_alive(engine, 0)

    pool.release(engine)

    assert engine.closed is True
    assert pool.loaded_ids() == []


def test_negative_keep_alive_never_expires():
    pool = ModelPool(
        {"a": "/a"},
        lambda alias, path: FakeEngine(alias),
        max_loaded=1,
        keep_alive=30,
        reap_interval=60,
    )
    engine, _ = pool.acquire("a")
    pool.keep_alive(engine, -1)
    pool.release(engine)

    assert pool.expire(now=time.monotonic() + 10**9) == []
    assert engine.closed is False
    assert pool.expiries()["a"] is None


def test_expiries_reports_the_remaining_ttl_as_an_iso_timestamp():
    pool = ModelPool(
        {"a": "/a"},
        lambda alias, path: FakeEngine(alias),
        max_loaded=1,
        keep_alive=60,
        reap_interval=60,
    )
    engine, _ = pool.acquire("a")
    pool.release(engine)

    expires = pool.expiries()["a"]
    assert expires is not None
    deadline = datetime.fromisoformat(expires.replace("Z", "+00:00"))
    assert deadline > datetime.now(UTC)


def test_pool_rejects_an_empty_model_set():
    with pytest.raises(ValueError):
        ModelPool({}, lambda alias, path: FakeEngine(alias))


def test_max_loaded_is_at_least_one():
    pool = ModelPool({"a": "/a", "b": "/b"}, lambda alias, path: FakeEngine(alias), max_loaded=0)

    pool.get("a")
    pool.get("b")

    assert pool.loaded_ids() == ["b"]
