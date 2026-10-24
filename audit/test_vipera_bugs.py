"""Bug-audit probes for Vipera 1.2.0.

Each test states the behavior the code should have. A failure is a confirmed
defect, not a broken test: run with

    .venv/bin/python -m pytest audit -q

This directory is outside `testpaths`, so the shipped suite is unaffected.
"""

from __future__ import annotations

import asyncio
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vipera import config
from vipera.commands import bench_cmd, chat_cmd, serve_cmd
from vipera.config import DEFAULT_CONFIG
from vipera.optimize.layer import OptimizeLayer
from vipera.optimize.shaping import estimate_tokens, shape
from vipera.optimize.tuning import TuningStore, apply_tuning
from vipera.planner import _KV_BYTES_PER_ELEM, Plan
from vipera.pool import ModelPool
from vipera.registry import ModelSpec
from vipera.server import ChatRequest, build_app

# --------------------------------------------------------------------------
# Fixtures / fakes
# --------------------------------------------------------------------------


class FakeEngine:
    """Engine that records calls and returns OpenAI-shaped bodies."""

    def __init__(self, *, context=512, usage=True):
        self.calls = []
        self.plan = types.SimpleNamespace(context=context)
        self.usage = usage

    def chat(self, messages, stream=False, **kw):
        self.calls.append({"messages": messages, "stream": stream, **kw})
        if stream:

            def _gen():
                yield {"choices": [{"delta": {"content": "hi"}}]}
                yield {"choices": [{"delta": {"content": " there"}}]}

            return _gen()
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }
            ]
        }
        if self.usage:
            body["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
        return body

    def tokenize(self, text):
        return [0] * len(str(text).split())

    def embed(self, texts):
        return [[float(len(t)), 0.5] for t in texts]

    def close(self):
        pass


def _hw(*, gpu="GTX 1650", vram=4.0, free=3.5, cores=4, ram=32.0):
    return types.SimpleNamespace(
        gpu_name=gpu,
        vram_total_gb=vram,
        vram_free_gb=free,
        cpu_cores=cores,
        ram_total_gb=ram,
    )


def _plan(**overrides) -> Plan:
    base = dict(
        gpu_layers=10,
        cpu_threads=4,
        context=8192,
        kv_cache="q8_0",
        flash_attn=True,
        n_batch=512,
        offload_kqv=True,
        use_mlock=False,
        note="Hybrid: 10/40 layers",
        n_layers=40,
        ram_estimate_gb=1.0,
    )
    base.update(overrides)
    return Plan(**base)


def _model(alias="m"):
    return ModelSpec(alias=alias, path=Path("/tmp/m.gguf"), meta={"n_layers": 40})


# --------------------------------------------------------------------------
# 1. Ollama: `format` (JSON mode) is parsed and then silently dropped
# --------------------------------------------------------------------------


def test_ollama_format_json_reaches_the_engine_as_response_format():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model", protocols=["ollama"]))

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "format": "json",
        },
    )

    assert r.status_code == 200
    assert engine.calls[-1].get("response_format") == {"type": "json_object"}


def test_ollama_generate_format_json_reaches_the_engine_as_response_format():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model", protocols=["ollama"]))

    client.post(
        "/api/generate",
        json={"model": "test-model", "prompt": "hi", "stream": False, "format": "json"},
    )

    assert engine.calls[-1].get("response_format") == {"type": "json_object"}


# --------------------------------------------------------------------------
# 2. Ollama: an empty `messages` list is forwarded instead of rejected
# --------------------------------------------------------------------------


def test_ollama_chat_rejects_empty_messages():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model", protocols=["ollama"]))

    r = client.post(
        "/api/chat",
        json={"model": "test-model", "messages": [], "stream": False},
    )

    assert r.status_code == 400
    assert engine.calls == [], "an empty conversation must not reach the model"


def test_ollama_chat_does_not_silently_discard_images():
    engine = FakeEngine()
    client = TestClient(build_app(engine, "test-model", protocols=["ollama"]))

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "what is in this picture?", "images": ["aGVsbG8="]}
            ],
            "stream": False,
        },
    )

    # A text-only model cannot answer; returning 200 pretends it saw the image.
    assert r.status_code in {400, 422}, "the image was accepted and then ignored"


# --------------------------------------------------------------------------
# 3. Ollama: non-streaming replies report zero token counts without usage
# --------------------------------------------------------------------------


def test_ollama_non_stream_counts_tokens_when_backend_reports_no_usage():
    engine = FakeEngine(usage=False)
    client = TestClient(build_app(engine, "test-model", protocols=["ollama"]))

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    body = r.json()
    assert body["eval_count"] > 0, "completion tokens are known from the text"
    assert body["prompt_eval_count"] > 0, "prompt tokens are known from the input"


# --------------------------------------------------------------------------
# 4. An engine failure returns a non-JSON 500 from both protocol layers
# --------------------------------------------------------------------------


class RaisingEngine:
    plan = types.SimpleNamespace(context=512)

    def chat(self, messages, stream=False, **kw):
        raise RuntimeError("model crashed")


def test_openai_engine_failure_returns_a_json_error_envelope():
    client = TestClient(build_app(RaisingEngine(), "test-model"), raise_server_exceptions=False)

    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 500
    assert r.headers["content-type"].startswith("application/json")
    assert "error" in r.json()


def test_ollama_engine_failure_returns_its_error_shape():
    client = TestClient(
        build_app(RaisingEngine(), "test-model", protocols=["ollama"]),
        raise_server_exceptions=False,
    )

    r = client.post(
        "/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert r.status_code == 500
    assert r.headers["content-type"].startswith("application/json")
    assert "error" in r.json()


# --------------------------------------------------------------------------
# 5. Shaping ignores the tool schema it is handed
# --------------------------------------------------------------------------


def test_shaping_accounts_for_the_tool_schema():
    engine = types.SimpleNamespace(plan=types.SimpleNamespace(context=512))
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "x" * 900},
    ]
    # ~100 tokens of schema: small enough that trimming the message can still
    # make the request fit, large enough that ignoring it would overflow 512.
    tools = [{"type": "function", "function": {"name": "f", "description": "d" * 300}}]
    layer = OptimizeLayer(shaping=True)

    assert layer.prepare(engine, messages, max_tokens=64).applied is False

    prepared = layer.prepare(engine, messages, max_tokens=64, tools=tools)

    # The schema is rendered into the prompt by the chat template; it cannot be
    # ignored when deciding whether the request fits.
    assert prepared.applied is True
    assert prepared.estimated_after <= prepared.budget_tokens


# --------------------------------------------------------------------------
# 6. Dropping history can orphan a tool message
# --------------------------------------------------------------------------


def test_shaping_never_leaves_a_tool_message_without_its_assistant_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q0 " + "x" * 300},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "t0 " + "z" * 300},
        {"role": "assistant", "content": "a0 " + "y" * 300},
        {"role": "user", "content": "q1 " + "x" * 300},
        {"role": "assistant", "content": "a1 " + "y" * 300},
        {"role": "user", "content": "final"},
    ]

    broken = []
    for budget in range(80, estimate_tokens(messages)):
        shaped = shape(messages, budget_tokens=budget, max_tokens=0).messages
        pending_calls = False
        for message in shaped:
            if message.get("role") == "assistant":
                pending_calls = bool(message.get("tool_calls"))
            elif message.get("role") == "tool":
                if not pending_calls:
                    broken.append((budget, shaped))
                    break
            elif message.get("role") == "user":
                pending_calls = False
        if broken:
            break

    assert not broken, f"orphaned tool message at budget {broken[0][0] if broken else '?'}"


# --------------------------------------------------------------------------
# 7. A measured split is replayed at a context it was never measured at
# --------------------------------------------------------------------------


def test_tuning_profile_is_not_replayed_at_a_larger_context(tmp_path):
    store = TuningStore(tmp_path / "tuning.json")
    hw = _hw()
    # Fastest run was measured at 512 context; serve now asks for 8192.
    store.record("m", hw, gpu_layers=20, ctx=512, gen_tps=9.0)

    out = apply_tuning(_plan(gpu_layers=10, context=8192), "m", hw, store)

    assert out.gpu_layers == 10, (
        "a 20-layer split measured at 512 context was replayed at 8192 context"
    )


# --------------------------------------------------------------------------
# 8. A lazy load blocks the event loop
# --------------------------------------------------------------------------


def test_pool_lazy_load_does_not_block_the_event_loop():
    import httpx

    class SlowLoadEngine(FakeEngine):
        pass

    def factory(alias, path):
        time.sleep(0.4)  # stand in for planning + loading a GGUF
        return SlowLoadEngine()

    pool = ModelPool({"a": "/a.gguf", "b": "/b.gguf"}, factory)
    app = build_app(pool=pool)
    transport = httpx.ASGITransport(app=app)

    async def scenario():
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:

            async def ticker(start):
                await asyncio.sleep(0.05)
                return time.perf_counter() - start

            start = time.perf_counter()
            tick = asyncio.create_task(ticker(start))
            chat = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={"model": "a", "messages": [{"role": "user", "content": "hi"}]},
                )
            )
            await chat
            delay = await tick
            return delay

    delay = asyncio.run(scenario())
    pool.close()

    assert delay < 0.2, f"a 50ms timer fired after {delay:.2f}s: the load blocked the loop"


# --------------------------------------------------------------------------
# 9. Eviction closes an engine while it is generating
# --------------------------------------------------------------------------


def test_pool_does_not_close_an_engine_that_is_mid_generation():
    engines = {}

    class BusyEngine:
        def __init__(self, name):
            self.name = name
            self.closed = False
            self.busy = False
            self.closed_while_busy = False
            self.plan = types.SimpleNamespace(context=512)

        def chat(self, messages, stream=False, **kw):
            self.busy = True
            time.sleep(0.3)  # generation in the worker thread
            if self.closed:
                self.closed_while_busy = True
            self.busy = False
            return {
                "choices": [{"message": {"role": "assistant", "content": self.name}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        def close(self):
            self.closed = True

    def factory(alias, path):
        engine = BusyEngine(alias)
        engines[alias] = engine
        return engine

    pool = ModelPool({"a": "/a.gguf", "b": "/b.gguf"}, factory, max_loaded=1)
    app = build_app(pool=pool)
    route = next(
        r.endpoint for r in app.routes if getattr(r, "path", None) == "/v1/chat/completions"
    )

    async def scenario():
        first = asyncio.create_task(
            route(ChatRequest(model="a", messages=[{"role": "user", "content": "hi"}]))
        )
        await asyncio.sleep(0.1)  # "a" is generating, lock held
        second = asyncio.create_task(
            route(ChatRequest(model="b", messages=[{"role": "user", "content": "hi"}]))
        )
        await second  # resolving "b" evicts and closes "a" immediately
        await first

    asyncio.run(scenario())
    pool.close()

    assert engines["a"].closed_while_busy is False, (
        "max_loaded=1 closed model 'a' while its request was still generating"
    )


# --------------------------------------------------------------------------
# 10. An invalid request loads a model before it is rejected
# --------------------------------------------------------------------------


def test_invalid_request_is_rejected_before_the_pool_loads_a_model():
    loads = []

    def factory(alias, path):
        loads.append(alias)
        return FakeEngine()

    pool = ModelPool({"a": "/a.gguf", "b": "/b.gguf"}, factory)
    client = TestClient(build_app(pool=pool))

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "a",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        },
    )

    assert r.status_code == 400
    assert loads == [], "a request rejected with 400 must not trigger a model load"


# --------------------------------------------------------------------------
# 11. A failed `serve start` deletes the running server's pid file
# --------------------------------------------------------------------------


def test_failed_serve_start_keeps_an_existing_pid_file(monkeypatch, tmp_path):
    monkeypatch.setattr(serve_cmd, "VIPERA_HOME", tmp_path)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"default_model": "m", "host": "127.0.0.1", "port": 8686, "ctx": 1024})
    monkeypatch.setattr(serve_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(serve_cmd, "get_model", lambda name: _model())
    monkeypatch.setattr(serve_cmd, "detect", lambda: _hw())
    monkeypatch.setattr(serve_cmd, "plan", lambda *a, **kw: _plan(context=1024))
    monkeypatch.setattr(
        serve_cmd, "create_engine", lambda p, path, backend="inprocess": FakeEngine()
    )
    monkeypatch.setattr("vipera.server.build_app", lambda engine, model, **kw: object())

    def _port_in_use(app, **kw):
        raise OSError("address already in use")

    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=_port_in_use))

    pid_file = tmp_path / "serve-8686.pid"
    pid_file.write_text("4242")  # a gateway that is already running

    with pytest.raises(OSError):
        serve_cmd.start(
            model="m", host="127.0.0.1", port=8686, profile="auto", ctx=1024, bench=False
        )

    assert pid_file.exists() and pid_file.read_text() == "4242", (
        "a failed second start erased the pid file of the running gateway"
    )


# --------------------------------------------------------------------------
# 12. Config validation gaps
# --------------------------------------------------------------------------


def test_config_rejects_an_unknown_profile_at_write_time():
    with pytest.raises(ValueError):
        config._coerce_strict({**DEFAULT_CONFIG, "profile": "turbo"})


def test_config_rejects_an_out_of_range_port_at_write_time():
    with pytest.raises(ValueError):
        config._coerce_strict({**DEFAULT_CONFIG, "port": 99999})


def test_config_warns_about_an_unparseable_bool(monkeypatch, capsys):
    cfg = config._coerce({"shaping": "ture"})

    assert cfg["shaping"] is True, "an invalid bool must fall back to the default"
    assert "shaping" in capsys.readouterr().err


# --------------------------------------------------------------------------
# 13. `profile` in config is ignored by chat and bench
# --------------------------------------------------------------------------


def test_chat_honours_the_configured_profile(monkeypatch):
    seen = {}
    cfg = {**DEFAULT_CONFIG, "default_model": "m", "profile": "cpu", "tuning": False}
    monkeypatch.setattr(chat_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(chat_cmd, "get_model", lambda name: _model(name))
    monkeypatch.setattr(chat_cmd, "detect", lambda: _hw())

    def _fake_plan(*args, **kwargs):
        seen.update(kwargs)
        return _plan()

    monkeypatch.setattr(chat_cmd, "plan", _fake_plan)
    monkeypatch.setattr(
        chat_cmd, "create_engine", lambda p, path, backend="inprocess": FakeEngine()
    )

    chat_cmd._build("m")

    assert seen.get("profile") == "cpu"


def test_bench_honours_the_configured_profile(monkeypatch):
    seen = {}
    cfg = {**DEFAULT_CONFIG, "default_model": "m", "profile": "cpu", "tuning": False}
    monkeypatch.setattr(bench_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(bench_cmd, "get_model", lambda name: _model(name))
    monkeypatch.setattr(
        bench_cmd,
        "detect",
        lambda: types.SimpleNamespace(cpu_cores=4, ram_free_gb=8.0, gpu_name="none"),
    )

    def _fake_plan(*args, **kwargs):
        seen.update(kwargs)
        return _plan(context=int(kwargs.get("ctx") or 512))

    monkeypatch.setattr(bench_cmd, "plan", _fake_plan)

    class Boom(FakeEngine):
        def __init__(self, *a, **kw):
            raise RuntimeError("no model file in this audit")

    monkeypatch.setattr(bench_cmd, "create_engine", lambda p, path, backend="inprocess": Boom())

    bench_cmd.run(model="m", ctx="512", max_tokens=2)

    assert seen.get("profile") == "cpu"


# --------------------------------------------------------------------------
# 14. IQ4_NL is priced below its real block size
# --------------------------------------------------------------------------


def test_iq4_nl_kv_estimate_uses_the_exact_block_size():
    # ggml IQ4_NL: 18 bytes per 32 weights.
    assert _KV_BYTES_PER_ELEM["iq4_nl"] == pytest.approx(18 / 32)
