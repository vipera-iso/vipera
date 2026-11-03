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
    # Compare as paths: str() of a Windows path uses backslashes.
    assert Path(seen["path"]) == Path("/tmp/m.gguf")
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

    monkeypatch.setattr(chat_cmd, "_attach", lambda *a, **k: None)
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

    monkeypatch.setattr(chat_cmd, "_attach", lambda *a, **k: None)
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
    monkeypatch.setattr(chat_cmd, "_attach", lambda *a, **k: None)
    monkeypatch.setattr(chat_cmd, "_build", lambda model: engine)
    monkeypatch.setattr(chat_cmd, "repl_loop", lambda *a, **kw: None)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))

    chat_cmd.repl(model="m", typewriter=False)

    assert engine.closed is True


def test_repl_uses_the_gateway_instead_of_loading_a_second_copy(monkeypatch):
    """A gateway already holds the weights; loading them again is the bug."""
    attached = FakeEngine()
    attached.base_url = "http://127.0.0.1:8686"

    def _no_local_load(model):
        raise AssertionError("loaded a second copy while a gateway was serving it")

    monkeypatch.setattr(chat_cmd, "_attach", lambda *a, **k: attached)
    monkeypatch.setattr(chat_cmd, "_build", _no_local_load)
    monkeypatch.setattr(chat_cmd, "repl_loop", lambda *a, **kw: None)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))

    chat_cmd.repl(model="m", typewriter=False)

    assert attached.closed is True


def test_repl_loads_locally_when_no_gateway_serves_the_model(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(chat_cmd, "_attach", lambda *a, **k: None)
    monkeypatch.setattr(chat_cmd, "_build", lambda model: engine)
    monkeypatch.setattr(chat_cmd, "repl_loop", lambda *a, **kw: None)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))

    chat_cmd.repl(model="m", typewriter=False)

    assert engine.closed is True


def test_attach_points_at_the_gateway_the_config_names(monkeypatch):
    seen = {}

    def fake_attach(url, model, *, api_key=None):
        seen.update(url=url, model=model, api_key=api_key)
        return None

    monkeypatch.setattr(chat_cmd.gateway, "attach", fake_attach)

    chat_cmd._attach("m", {"host": "0.0.0.0", "port": 9999, "api_key": "k"})

    # A wildcard bind is where serve listens, not an address to dial.
    assert seen == {"url": "http://127.0.0.1:9999", "model": "m", "api_key": "k"}


def test_local_flag_skips_the_gateway(monkeypatch):
    """`--local` is an answer, not a hint: do not even probe the gateway."""
    engine = FakeEngine()
    probes = []
    monkeypatch.setattr(chat_cmd, "_attach", lambda *a, **k: probes.append(a) or None)
    monkeypatch.setattr(chat_cmd, "_build", lambda model: engine)
    monkeypatch.setattr(chat_cmd, "repl_loop", lambda *a, **kw: None)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))

    chat_cmd.repl(model="m", typewriter=False, local=True)

    assert probes == []
    assert engine.closed is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:9000", "http://127.0.0.1:9000"),
        ("http://gpu-box:9000/", "http://gpu-box:9000"),
        ("127.0.0.1:9000", "http://127.0.0.1:9000"),
        ("9000", "http://127.0.0.1:9000"),
    ],
)
def test_gateway_flag_takes_what_a_person_pastes(value, expected):
    # A gateway started with `--port 9000` is not reachable at the default one.
    assert chat_cmd._gateway_url({}, value) == expected


def test_gateway_flag_falls_back_to_the_config():
    cfg = {"host": "0.0.0.0", "port": 8686}
    assert chat_cmd._gateway_url(cfg) == "http://127.0.0.1:8686"


def test_attach_probes_the_gateway_the_flag_names(monkeypatch):
    seen = {}

    def fake_attach(url, model, *, api_key=None):
        seen.update(url=url, model=model, api_key=api_key)
        return None

    monkeypatch.setattr(chat_cmd.gateway, "attach", fake_attach)

    chat_cmd._attach("m", {"port": 8686, "api_key": "k"}, gateway_url="127.0.0.1:9000")

    assert seen == {"url": "http://127.0.0.1:9000", "model": "m", "api_key": "k"}


def _wire_build(monkeypatch, cfg, on_engine):
    monkeypatch.setattr(chat_cmd, "get_model", _spec)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: dict(cfg))
    monkeypatch.setattr(chat_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(chat_cmd, "plan", lambda *a, **kw: _plan())

    def fake_create(p, path, backend="inprocess"):
        on_engine(p, path)
        return object()

    monkeypatch.setattr(chat_cmd, "create_engine", fake_create)


def test_build_uses_the_configured_planner_profile(monkeypatch):
    """`profile` in config must reach the planner here too, not just in serve."""
    seen = {}
    monkeypatch.setattr(chat_cmd, "get_model", _spec)
    monkeypatch.setattr(chat_cmd, "load_config", lambda: {**DEFAULT_CONFIG, "profile": "cpu"})
    monkeypatch.setattr(chat_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(chat_cmd, "apply_tuning", lambda p, alias, hw: p)

    def fake_plan(*args, **kwargs):
        seen.update(kwargs)
        return _plan()

    monkeypatch.setattr(chat_cmd, "plan", fake_plan)
    monkeypatch.setattr(chat_cmd, "create_engine", lambda p, path, backend="inprocess": object())

    chat_cmd._build("m")

    assert seen["profile"] == "cpu"


def test_build_applies_a_measured_tuning_profile(monkeypatch):
    seen = {}
    called = []

    def fake_tuning(p, alias, hw):
        called.append(alias)
        return p.__class__(**{**p.__dict__, "gpu_layers": 3})

    monkeypatch.setattr(chat_cmd, "apply_tuning", fake_tuning)
    _wire_build(monkeypatch, DEFAULT_CONFIG, lambda p, path: seen.update(plan=p))

    chat_cmd._build("m")

    assert called == ["m"]
    assert seen["plan"].gpu_layers == 3


def test_build_names_the_server_holding_the_card(monkeypatch, capsys):
    """A CPU-only plan on a machine with a card deserves the reason.

    The planner only sees `vram_free_gb`, so a second server reads as "this card
    is small", not "this card is busy" - on gpt-oss-120b that is 0.6 t/s
    against 1.8, with no hint about why.
    """
    _wire_build(monkeypatch, DEFAULT_CONFIG, lambda p, path: None)
    monkeypatch.setattr(
        chat_cmd,
        "detect",
        lambda: types.SimpleNamespace(vram_total_gb=4.0, vram_free_gb=0.12),
    )
    monkeypatch.setattr(
        chat_cmd, "_other_servers", lambda: [("50973", "llama-server --port 50973")]
    )

    chat_cmd._build("m")

    out = capsys.readouterr().out
    assert "port 50973" in out
    assert "3.9 of 4.0 GB" in out
    assert "vipera serve stop" in out


def test_build_stays_quiet_when_the_card_is_free(monkeypatch, capsys):
    _wire_build(monkeypatch, DEFAULT_CONFIG, lambda p, path: None)
    monkeypatch.setattr(
        chat_cmd,
        "detect",
        lambda: types.SimpleNamespace(vram_total_gb=4.0, vram_free_gb=3.64),
    )
    monkeypatch.setattr(chat_cmd, "_other_servers", lambda: [])

    chat_cmd._build("m")

    assert "another llama-server" not in capsys.readouterr().out


def test_build_does_not_explain_a_plan_that_uses_the_card(monkeypatch, capsys):
    _wire_build(monkeypatch, DEFAULT_CONFIG, lambda p, path: None)
    monkeypatch.setattr(
        chat_cmd,
        "plan",
        lambda *a, **kw: Plan(**{**_plan().__dict__, "gpu_layers": 7}),
    )
    monkeypatch.setattr(
        chat_cmd,
        "detect",
        lambda: types.SimpleNamespace(vram_total_gb=4.0, vram_free_gb=0.12),
    )
    monkeypatch.setattr(
        chat_cmd, "_other_servers", lambda: [("50973", "llama-server --port 50973")]
    )

    chat_cmd._build("m")

    assert "another llama-server" not in capsys.readouterr().out


def test_other_servers_names_running_children_only(monkeypatch):
    class Proc:
        def __init__(self, cmdline):
            self.info = {"cmdline": cmdline}

    procs = [
        Proc(["/home/u/.vipera/bin/llama-server", "-m", "m.gguf", "--port", "50973"]),
        Proc(["/usr/bin/python", "-m", "vipera", "serve", "start"]),
        Proc(["/home/u/.vipera/bin/llama-server", "--help"]),
    ]
    monkeypatch.setattr(chat_cmd.psutil, "process_iter", lambda attrs=None: procs)

    assert [port for port, _ in chat_cmd._other_servers()] == ["50973"]


def test_build_skips_tuning_when_the_config_turns_it_off(monkeypatch):
    called = []

    def fake_tuning(p, alias, hw):
        called.append(alias)
        return p

    monkeypatch.setattr(chat_cmd, "apply_tuning", fake_tuning)
    _wire_build(monkeypatch, {**DEFAULT_CONFIG, "tuning": False}, lambda p, path: None)

    chat_cmd._build("m")

    assert called == []
