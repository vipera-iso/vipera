import sys
import types

import psutil
import pytest
import typer

from vipera import config
from vipera.commands import serve_cmd
from vipera.planner import Plan


def _plan(**overrides):
    base = dict(
        gpu_layers=4,
        cpu_threads=3,
        context=1024,
        kv_cache="q8_0",
        flash_attn=True,
        n_batch=128,
        offload_kqv=True,
        use_mlock=False,
        note="Hybrid",
        ram_estimate_gb=1.0,
    )
    base.update(overrides)
    return Plan(**base)


def _cfg(**overrides):
    cfg = dict(config.DEFAULT_CONFIG)
    cfg.update({"default_model": "m", "host": "127.0.0.1", "port": 8686, "ctx": 1024})
    cfg.update(overrides)
    return cfg


def _model():
    return {"name": "m", "path": "/tmp/m.gguf", "meta": {"n_layers": 8, "quant": "Q4_K_M"}}


class FakeEngine:
    def __init__(self, plan, path):
        self.plan = plan
        self.path = path
        self.closed = False
        self.chat_calls = []

    def chat(self, messages, stream=False, **kw):
        self.chat_calls.append({"messages": messages, "stream": stream, **kw})
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 20},
        }

    def close(self):
        self.closed = True


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setattr(serve_cmd, "VIPERA_HOME", tmp_path)
    return tmp_path


def test_resolve_option_unwraps_typer_optioninfo():
    option = typer.Option("from-option")
    assert serve_cmd._resolve_option(option, "fallback") == "fallback"
    assert serve_cmd._resolve_option(None, "fallback") == "fallback"
    assert serve_cmd._resolve_option("explicit", "fallback") == "explicit"


def test_quick_bench_reports_speed(capsys):
    engine = FakeEngine(_plan(), "/tmp/m.gguf")
    serve_cmd._quick_bench(engine)
    assert len(engine.chat_calls) == 1
    assert engine.chat_calls[0]["max_tokens"] == 20


def test_quick_bench_swallows_engine_errors(monkeypatch):
    class Broken(FakeEngine):
        def chat(self, *args, **kwargs):
            raise RuntimeError("no model")

    serve_cmd._quick_bench(Broken(_plan(), "/tmp/m.gguf"))


def test_start_serves_without_running_the_benchmark(monkeypatch, isolated_home):
    calls = {}

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(serve_cmd, "Engine", FakeEngine)
    monkeypatch.setattr(serve_cmd, "_quick_bench", lambda engine: calls.setdefault("bench", True))
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda app, **kw: calls.setdefault("uvicorn", kw)),
    )
    monkeypatch.setattr(
        "vipera.server.build_app", lambda engine, model: calls.setdefault("app", (engine, model))
    )

    serve_cmd.start(model="m", host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=False)

    # Readiness must not wait on a speed check.
    assert "bench" not in calls
    assert calls["uvicorn"]["host"] == "127.0.0.1"
    assert calls["uvicorn"]["port"] == 8686
    engine = calls["app"][0]
    assert engine.closed is True
    assert not (isolated_home / "serve-8686.pid").exists()


def test_start_runs_bench_only_when_requested(monkeypatch, isolated_home):
    calls = {}

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(serve_cmd, "Engine", FakeEngine)
    monkeypatch.setattr(
        serve_cmd, "_quick_bench", lambda engine: calls.setdefault("bench_calls", 0)
    )
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: None))
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model: object())

    serve_cmd.start(model="m", host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=True)

    assert "bench_calls" in calls


def test_start_requires_a_model(monkeypatch):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg(default_model=None))
    with pytest.raises(typer.Exit) as exc:
        serve_cmd.start(
            model=None, host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=False
        )
    assert exc.value.exit_code == 1


def test_start_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: None)
    with pytest.raises(typer.Exit) as exc:
        serve_cmd.start(
            model="missing",
            host="127.0.0.1",
            port=8686,
            profile="auto",
            ctx=1024,
            bench=False,
        )
    assert exc.value.exit_code == 1


def test_start_rejects_invalid_profile(monkeypatch):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())

    def _bad_plan(*args, **kwargs):
        raise ValueError("unknown profile 'turbo'")

    monkeypatch.setattr(serve_cmd, "plan", _bad_plan)

    with pytest.raises(typer.Exit) as exc:
        serve_cmd.start(
            model="m", host="127.0.0.1", port=8686, profile="turbo", ctx=1024, bench=False
        )
    assert exc.value.exit_code == 1


def test_status_reports_online_and_offline(monkeypatch):
    class Sock:
        def __init__(self, code):
            self.code = code

        def connect_ex(self, address):
            return self.code

        def close(self):
            pass

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd.socket, "socket", lambda *a, **kw: Sock(0))
    serve_cmd.status()

    monkeypatch.setattr(serve_cmd.socket, "socket", lambda *a, **kw: Sock(111))
    serve_cmd.status()


def test_stop_without_pid_file_exits(monkeypatch, isolated_home):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    with pytest.raises(typer.Exit) as exc:
        serve_cmd.stop(port=8686, force=False, yes=True)
    assert exc.value.exit_code == 1


def test_stop_removes_unreadable_pid_file(monkeypatch, isolated_home):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    pid_file = isolated_home / "serve-8686.pid"
    pid_file.write_text("not-a-pid")

    with pytest.raises(typer.Exit) as exc:
        serve_cmd.stop(port=8686, force=False, yes=True)

    assert exc.value.exit_code == 1
    assert not pid_file.exists()


def test_stop_removes_stale_pid_file(monkeypatch, isolated_home):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    pid_file = isolated_home / "serve-8686.pid"
    pid_file.write_text("123")

    def _gone(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(serve_cmd.psutil, "Process", _gone)

    with pytest.raises(typer.Exit) as exc:
        serve_cmd.stop(port=8686, force=False, yes=True)

    assert exc.value.exit_code == 1
    assert not pid_file.exists()


class FakeProcess:
    def __init__(self, cmdline=("vipera", "serve"), timeout_once=False):
        self._cmdline = cmdline
        self.timeout_once = timeout_once
        self.terminated = False
        self.killed = False
        self.waits = 0

    def cmdline(self):
        return list(self._cmdline)

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waits += 1
        if self.timeout_once and self.waits == 1:
            raise psutil.TimeoutExpired(timeout)
        return 0

    def kill(self):
        self.killed = True


def test_stop_refuses_unrelated_process(monkeypatch, isolated_home):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    pid_file = isolated_home / "serve-8686.pid"
    pid_file.write_text("123")
    proc = FakeProcess(cmdline=("nginx", "-g", "daemon off;"))
    monkeypatch.setattr(serve_cmd.psutil, "Process", lambda pid: proc)

    with pytest.raises(typer.Exit) as exc:
        serve_cmd.stop(port=8686, force=False, yes=True)

    assert exc.value.exit_code == 1
    assert proc.terminated is False
    assert pid_file.exists()


def test_stop_terminates_the_gateway(monkeypatch, isolated_home):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    pid_file = isolated_home / "serve-8686.pid"
    pid_file.write_text("123")
    proc = FakeProcess()
    monkeypatch.setattr(serve_cmd.psutil, "Process", lambda pid: proc)

    serve_cmd.stop(port=8686, force=False, yes=True)

    assert proc.terminated is True
    assert proc.killed is False
    assert not pid_file.exists()


def test_stop_force_kills_after_timeout(monkeypatch, isolated_home):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    pid_file = isolated_home / "serve-8686.pid"
    pid_file.write_text("123")
    proc = FakeProcess(timeout_once=True)
    monkeypatch.setattr(serve_cmd.psutil, "Process", lambda pid: proc)

    serve_cmd.stop(port=8686, force=True, yes=True)

    assert proc.terminated is True
    assert proc.killed is True
    assert not pid_file.exists()
