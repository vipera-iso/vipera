"""A backend installed as a plugin must be selectable by name."""

import types

import pytest

from vipera import config as cfg_mod
from vipera.backends import names as names_mod
from vipera.backends import registry
from vipera.errors import BackendError
from vipera.planner import Plan


def _plan():
    return Plan(
        gpu_layers=0,
        cpu_threads=2,
        context=512,
        kv_cache="q8_0",
        flash_attn=False,
        n_batch=64,
        offload_kqv=False,
        use_mlock=False,
        note="CPU ONLY",
    )


class FakePluginEngine:
    def __init__(self, plan, path):
        self.plan = plan
        self.path = path

    def chat(self, messages, stream=False, **kw):
        return {"choices": [{"message": {"role": "assistant", "content": "plugin"}}]}

    def tokenize(self, text):
        return []

    def close(self):
        return None


def test_create_engine_calls_the_registered_factory(monkeypatch):
    monkeypatch.setattr(
        registry,
        "_load_plugin",
        lambda name: lambda plan, path: FakePluginEngine(plan, path),
    )

    engine = registry.create_engine(_plan(), "/tmp/m.gguf", backend="vllm")

    assert isinstance(engine, FakePluginEngine)
    assert engine.path == "/tmp/m.gguf"


def test_unknown_backend_raises_a_typed_error(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: types.SimpleNamespace(select=lambda group: [])
    )

    with pytest.raises(BackendError) as exc:
        registry._load_plugin("vllm")

    assert "vllm" in str(exc.value)
    assert "inprocess" in str(exc.value)


def test_plugin_names_are_added_to_the_known_set(monkeypatch):
    class Entry:
        name = "vllm"

        @staticmethod
        def load():
            return None

    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: types.SimpleNamespace(select=lambda group: [Entry()]),
    )
    names_mod.plugin_names.cache_clear()

    try:
        assert "vllm" in names_mod.names()
        assert names_mod.is_known("vllm")
        assert not names_mod.is_known("nope")
    finally:
        names_mod.plugin_names.cache_clear()


def test_config_accepts_a_known_plugin_name(monkeypatch):
    monkeypatch.setattr(cfg_mod, "_is_known_backend", lambda name: name in {"inprocess", "vllm"})

    assert cfg_mod._coerce({"backend": "vllm"})["backend"] == "vllm"
    assert cfg_mod._coerce({"backend": "typo"})["backend"] == "inprocess"
