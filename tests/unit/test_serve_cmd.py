import signal
import sys
import time
import types
from pathlib import Path

import psutil
import pytest
import typer

from vipera import config
from vipera.commands import serve_cmd
from vipera.planner import Plan
from vipera.registry import ModelSpec


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
    return ModelSpec(alias="m", path=Path("/tmp/m.gguf"), meta={"n_layers": 8, "quant": "Q4_K_M"})


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


def test_run_server_unwinds_on_sigterm_and_restores_the_handler():
    # Without this, SIGTERM kills the process before the `finally` in start()
    # runs and a `backend: server` child keeps the model loaded.
    seen = {}

    class FakeUvicorn:
        @staticmethod
        def run(app, **kw):
            seen["handler"] = signal.getsignal(signal.SIGTERM)
            try:
                seen["handler"](signal.SIGTERM, None)
            except SystemExit as exc:
                seen["code"] = exc.code
                raise

    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit):
        serve_cmd._run_server(FakeUvicorn, object(), "127.0.0.1", 8686)

    assert seen["code"] == 0
    assert seen["handler"] is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_start_closes_the_engine_when_the_server_is_terminated(monkeypatch, isolated_home):
    engines = []

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())

    def _create(plan, path, backend="inprocess"):
        engine = FakeEngine(plan, path)
        engines.append(engine)
        return engine

    monkeypatch.setattr(serve_cmd, "create_engine", _create)
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model, **kw: object())

    def _terminated(app, **kw):
        # What the SIGTERM handler does inside uvicorn.run().
        raise SystemExit(0)

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=_terminated))

    with pytest.raises(SystemExit):
        serve_cmd.start(
            model="m",
            host="127.0.0.1",
            port=8686,
            profile="auto",
            ctx=1024,
            bench=False,
        )

    assert engines and engines[0].closed is True, "the model must be released"
    assert not (isolated_home / "serve-8686.pid").exists()


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
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": FakeEngine(plan, path),
    )
    monkeypatch.setattr(serve_cmd, "_quick_bench", lambda engine: calls.setdefault("bench", True))
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda app, **kw: calls.setdefault("uvicorn", kw)),
    )
    monkeypatch.setattr(
        "vipera.server.build_app",
        lambda engine, model, **kw: calls.setdefault("app", (engine, model)),
    )

    serve_cmd.start(model="m", host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=False)

    # Readiness must not wait on a speed check.
    assert "bench" not in calls
    assert calls["uvicorn"]["host"] == "127.0.0.1"
    assert calls["uvicorn"]["port"] == 8686
    engine = calls["app"][0]
    assert engine.closed is True
    assert not (isolated_home / "serve-8686.pid").exists()


def test_failed_start_keeps_the_running_gateways_pid_file(monkeypatch, isolated_home):
    # A second `serve start` that cannot bind must not erase the file of the
    # gateway that is already serving: `serve stop` would then report OFFLINE.
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": FakeEngine(plan, path),
    )
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model, **kw: object())

    def _port_in_use(app, **kw):
        # What uvicorn really does when the port is taken: it logs the OSError
        # and calls sys.exit(STARTUP_FAILURE), so start() sees a SystemExit.
        raise SystemExit(3)

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=_port_in_use))
    pid_file = isolated_home / "serve-8686.pid"
    pid_file.write_text("4242")

    with pytest.raises(SystemExit):
        serve_cmd.start(
            model="m", host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=False
        )

    assert pid_file.read_text() == "4242"


def _start(monkeypatch, *, cfg=None, **kwargs):
    """Run serve start with everything below the HTTP layer stubbed out."""
    seen = {}
    monkeypatch.setattr(serve_cmd, "load_config", lambda: cfg or _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": FakeEngine(plan, path),
    )
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: None))

    def _build_app(engine, model, *, pool=None, protocols=None):
        seen["protocols"] = protocols
        return object()

    monkeypatch.setattr("vipera.server.build_app", _build_app)
    options = dict(model="m", host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=False)
    options.update(kwargs)
    serve_cmd.start(**options)
    return seen


def test_start_mounts_only_openai_by_default(monkeypatch, isolated_home):
    assert _start(monkeypatch)["protocols"] == ["openai"]


def test_start_mounts_the_protocols_from_the_config(monkeypatch, isolated_home):
    seen = _start(monkeypatch, cfg=_cfg(protocols="openai,ollama"))
    assert seen["protocols"] == ["openai", "ollama"]


def test_start_protocol_option_overrides_the_config(monkeypatch, isolated_home):
    seen = _start(monkeypatch, cfg=_cfg(protocols="openai"), protocol="ollama")
    assert seen["protocols"] == ["ollama"]


def test_start_rejects_an_unknown_protocol_before_loading(monkeypatch, isolated_home):
    loaded = []
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(
        serve_cmd, "create_engine", lambda plan, path, backend="inprocess": loaded.append(path)
    )

    with pytest.raises(typer.Exit) as exc:
        serve_cmd.start(
            model="m",
            host="127.0.0.1",
            port=8686,
            profile="auto",
            ctx=1024,
            bench=False,
            protocol="anthropic",
        )

    assert exc.value.exit_code == 1
    assert loaded == [], "validation must happen before the model is loaded"


def test_start_runs_bench_only_when_requested(monkeypatch, isolated_home):
    calls = {}

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": FakeEngine(plan, path),
    )
    monkeypatch.setattr(
        serve_cmd, "_quick_bench", lambda engine: calls.setdefault("bench_calls", 0)
    )
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: None))
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model, **kw: object())

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


def test_is_loopback_only_accepts_loopback():
    # 0.0.0.0 is the canonical "expose to everything" bind; it must warn.
    assert serve_cmd._is_loopback("127.0.0.1")
    assert serve_cmd._is_loopback("127.0.0.53")
    assert serve_cmd._is_loopback("localhost")
    assert serve_cmd._is_loopback("::1")
    assert not serve_cmd._is_loopback("0.0.0.0")
    assert not serve_cmd._is_loopback("::")
    assert not serve_cmd._is_loopback("192.168.1.10")
    assert not serve_cmd._is_loopback("example.lan")


def test_start_warns_for_non_loopback_host_and_still_serves(monkeypatch, isolated_home, capsys):
    calls = {}
    slept = []

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": FakeEngine(plan, path),
    )
    monkeypatch.setattr(time, "sleep", slept.append)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda app, **kw: calls.setdefault("uvicorn", kw)),
    )
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model, **kw: object())

    serve_cmd.start(model="m", host="0.0.0.0", port=8686, profile="auto", ctx=1024, bench=False)

    out = capsys.readouterr().out
    assert "exposes the model to the network" in out
    # The grace message must describe what actually happens: Ctrl+C aborts,
    # the server keeps starting.
    assert "Serving in 5s" in out
    assert "Aborting in 5s" not in out
    assert slept == [5]
    assert calls["uvicorn"]["host"] == "0.0.0.0"


def test_start_grace_period_aborts_on_ctrl_c(monkeypatch, isolated_home):
    def boom(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg())
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(time, "sleep", boom)

    with pytest.raises(typer.Exit) as exc:
        serve_cmd.start(model="m", host="0.0.0.0", port=8686, profile="auto", ctx=1024, bench=False)

    assert exc.value.exit_code == 1


def test_start_passes_the_configured_backend_to_the_engine(monkeypatch, isolated_home):
    seen = {}

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg(backend="server"))
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": (
            seen.setdefault("backend", backend),
            FakeEngine(plan, path),
        )[1],
    )
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: None))
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model, **kw: object())

    serve_cmd.start(model="m", host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=False)

    assert seen["backend"] == "server"


def test_start_uses_the_configured_profile(monkeypatch, isolated_home):
    """`profile` in config used to be dead: serve always passed "auto"."""
    seen = {}

    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg(profile="cpu"))
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())

    def _fake_plan(*args, **kwargs):
        seen["profile"] = kwargs.get("profile")
        return _plan()

    monkeypatch.setattr(serve_cmd, "plan", _fake_plan)
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": FakeEngine(plan, path),
    )
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: None))
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model, **kw: object())

    serve_cmd.start(model="m", host="127.0.0.1", port=8686, profile=None, ctx=1024, bench=False)

    assert seen["profile"] == "cpu"


def test_start_with_models_builds_a_lazy_pool(monkeypatch, isolated_home):
    seen = {}
    created = []

    monkeypatch.setattr(
        serve_cmd, "load_config", lambda: _cfg(models=["m1", "m2"], default_model="m1")
    )
    monkeypatch.setattr(
        serve_cmd,
        "get_model",
        lambda name: ModelSpec(
            alias=name, path=Path(f"/tmp/{name}.gguf"), meta={"n_layers": 8, "quant": "Q4_K_M"}
        ),
    )
    monkeypatch.setattr(serve_cmd, "detect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        serve_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": (
            created.append(path),
            FakeEngine(plan, path),
        )[1],
    )
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: None))

    def _build_app(engine, model, *, pool=None, protocols=None):
        seen["engine"] = engine
        seen["pool"] = pool
        seen["protocols"] = protocols
        return object()

    monkeypatch.setattr("vipera.server.build_app", _build_app)

    serve_cmd.start(
        model=None,
        host="127.0.0.1",
        port=8686,
        profile="auto",
        ctx=1024,
        bench=False,
        models="m1,m2",
        max_loaded=2,
    )

    assert seen["engine"] is None
    assert seen["pool"] is not None
    assert seen["pool"].ids() == ["m1", "m2"]
    assert created == [], "models must load on first request, not at startup"


def test_start_rejects_an_unknown_alias_in_models(monkeypatch, isolated_home):
    monkeypatch.setattr(serve_cmd, "load_config", lambda: _cfg(models=["m1", "ghost"]))
    monkeypatch.setattr(
        serve_cmd,
        "get_model",
        lambda name: None if name == "ghost" else ModelSpec(alias=name, path=Path("/tmp/m.gguf")),
    )
    monkeypatch.setattr(
        serve_cmd, "list_models", lambda: [ModelSpec(alias="m1", path=Path("/tmp/m.gguf"))]
    )

    with pytest.raises(typer.Exit) as exc:
        serve_cmd.start(
            model=None,
            host="127.0.0.1",
            port=8686,
            profile="auto",
            ctx=1024,
            bench=False,
            models=None,
            max_loaded=2,
        )

    assert exc.value.exit_code == 1


def test_announce_plan_reports_the_tensor_split_not_experts(capsys):
    """A dense tensor split has no experts: the MoE row would read 41/40.

    `requires_server` is true for both plans, so the banner has to branch on
    which flag the plan carries.
    """
    p = _plan(
        n_layers=40,
        n_cpu_ffn=35,
        note="Tensor split: 5/40 dense FFN layers on GPU, attention on GPU",
    )

    serve_cmd._announce_plan("m", _model(), p, "server")
    out = capsys.readouterr().out

    assert "dense ffn" in out
    assert "5/40 layers on GPU, attention on GPU" in out
    assert "expert" not in out.lower()
    assert "41/40" not in out


def test_announce_plan_still_reports_experts_for_a_moe_plan(capsys):
    p = _plan(n_layers=24, gpu_layers=14, moe_cpu_layers=12, n_ubatch=896, note="MoE")

    serve_cmd._announce_plan("m", _model(), p, "server")
    out = capsys.readouterr().out

    assert "moe experts" in out
    assert "12/24 layers" in out
    assert "dense ffn" not in out
