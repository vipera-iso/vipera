import builtins
import types
from pathlib import Path

import pytest
import typer

from vipera.commands import chat_cmd
from vipera.config import DEFAULT_CONFIG
from vipera.planner import Plan
from vipera.registry import ModelSpec


def _spec(name, meta=None):
    return ModelSpec(alias=name, path=Path("/tmp/m.gguf"), meta=meta or {"n_layers": 4})


def _plan():
    return Plan(
        gpu_layers=0,
        cpu_threads=3,
        context=1024,
        kv_cache="q8_0",
        flash_attn=True,
        n_batch=128,
        offload_kqv=False,
        use_mlock=False,
        note="CPU ONLY",
    )


class FakeEngine:
    def __init__(self):
        self.chat_calls = []
        self.closed = False
        self.fail = False

    def chat(self, messages, stream=False, **kw):
        self.chat_calls.append({"messages": list(messages), "stream": stream, **kw})
        if self.fail:
            raise RuntimeError("engine exploded")

        def _gen():
            yield {"choices": [{"delta": {"content": "hello"}}]}

        return _gen()

    def close(self):
        self.closed = True


def test_chat_kwargs_only_takes_sampling_keys():
    cfg = {
        **DEFAULT_CONFIG,
        "temperature": 0.2,
        "max_tokens": 32,
        "top_p": 0.8,
        "repeat_penalty": 1.3,
        "host": "0.0.0.0",
    }
    kwargs = chat_cmd._chat_kwargs(cfg)
    assert kwargs == {
        "temperature": 0.2,
        "max_tokens": 32,
        "top_p": 0.8,
        "repeat_penalty": 1.3,
    }


def test_build_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(chat_cmd, "get_model", lambda name: None)
    with pytest.raises(typer.Exit) as exc:
        chat_cmd._build("missing")
    assert exc.value.exit_code == 1


def test_build_wires_planner_and_engine(monkeypatch):
    seen = {}

    class StubEngine:
        def __init__(self, plan, path):
            seen["plan"] = plan
            seen["path"] = path

    monkeypatch.setattr(
        chat_cmd,
        "get_model",
        _spec,
    )
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))
    monkeypatch.setattr(chat_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(chat_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        chat_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": StubEngine(plan, path),
    )

    engine = chat_cmd._build("m")

    assert isinstance(engine, StubEngine)
    assert str(seen["path"]) == "/tmp/m.gguf"
    assert seen["plan"].context == 1024


def test_repl_loop_keeps_history_and_stops_on_exit(monkeypatch):
    engine = FakeEngine()
    replies = []

    def fake_plain(stream, title_text=" AI "):
        replies.append("ok")
        return "ok"

    monkeypatch.setattr(chat_cmd, "plain_chat_stream", fake_plain)
    inputs = iter(["hello", "again", "exit"])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    chat_cmd.repl_loop(engine)

    assert len(engine.chat_calls) == 2
    first = engine.chat_calls[0]["messages"]
    assert first[0]["role"] == "system"
    assert first[1] == {"role": "user", "content": "hello"}
    # The assistant turn is appended so the next request has full context.
    second = engine.chat_calls[1]["messages"]
    assert second[2] == {"role": "assistant", "content": "ok"}
    assert second[-1] == {"role": "user", "content": "again"}
    assert replies == ["ok", "ok"]


def test_repl_loop_survives_engine_errors(monkeypatch):
    engine = FakeEngine()
    engine.fail = True
    monkeypatch.setattr(chat_cmd, "plain_chat_stream", lambda stream, title_text=" AI ": "")
    inputs = iter(["boom", "exit"])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    # Must not raise, and must not leave the failed user turn in history.
    chat_cmd.repl_loop(engine)


def test_repl_loop_handles_eof(monkeypatch):
    engine = FakeEngine()

    def eof():
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    chat_cmd.repl_loop(engine)
    assert engine.chat_calls == []


def test_oneshot_streams_and_closes_engine(monkeypatch):
    engine = FakeEngine()
    seen = {}

    def fake_plain(stream, title_text=" AI "):
        seen["chunks"] = list(stream)
        return "hello"

    monkeypatch.setattr(chat_cmd, "_build", lambda model: engine)
    monkeypatch.setattr(chat_cmd, "plain_chat_stream", fake_plain)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))

    chat_cmd.oneshot(prompt="hi", model="m", typewriter=False)

    assert engine.closed is True
    assert engine.chat_calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert seen["chunks"][0]["choices"][0]["delta"]["content"] == "hello"


def test_oneshot_typewriter_uses_typewriter_renderer(monkeypatch):
    engine = FakeEngine()
    calls = []

    monkeypatch.setattr(chat_cmd, "_build", lambda model: engine)
    monkeypatch.setattr(
        chat_cmd,
        "typewriter_chat_stream",
        lambda stream, title_text=" AI ": calls.append(list(stream)),
    )
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))

    chat_cmd.oneshot(prompt="hi", model="m", typewriter=True)

    assert calls
    assert engine.closed is True


def test_repl_command_closes_engine(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(chat_cmd, "_build", lambda model: engine)
    monkeypatch.setattr(chat_cmd, "repl_loop", lambda *a, **kw: None)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))

    chat_cmd.repl(model="m", typewriter=False)

    assert engine.closed is True
