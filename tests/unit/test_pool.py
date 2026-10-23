import pytest

from vipera.pool import ModelPool, UnknownModelError


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


def test_close_closes_every_loaded_engine(pool):
    a, _ = pool.get("a")
    b, _ = pool.get("b")

    pool.close()

    assert a.closed and b.closed
    assert pool.loaded_ids() == []


def test_pool_rejects_an_empty_model_set():
    with pytest.raises(ValueError):
        ModelPool({}, lambda alias, path: FakeEngine(alias))


def test_max_loaded_is_at_least_one():
    pool = ModelPool({"a": "/a", "b": "/b"}, lambda alias, path: FakeEngine(alias), max_loaded=0)

    pool.get("a")
    pool.get("b")

    assert pool.loaded_ids() == ["b"]
