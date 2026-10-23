import types
from pathlib import Path

import pytest
import typer

from vipera.commands import bench_cmd
from vipera.config import DEFAULT_CONFIG
from vipera.planner import Plan
from vipera.registry import ModelSpec


def _spec(name, meta=None):
    return ModelSpec(alias=name, path=Path("/tmp/m.gguf"), meta=meta or {"n_layers": 4})


def _plan(**overrides):
    base = dict(
        gpu_layers=2,
        cpu_threads=3,
        context=512,
        kv_cache="q8_0",
        flash_attn=True,
        n_batch=128,
        offload_kqv=True,
        use_mlock=False,
        note="Hybrid",
    )
    base.update(overrides)
    return Plan(**base)


class FakeLLM:
    def tokenize(self, data):
        return list(range(10))


class FakeEngine:
    def __init__(self, plan, path):
        self.plan = plan
        self.path = path
        self.llm = FakeLLM()
        self.closed = False
        self.chat_calls = []

    def tokenize(self, text):
        return self.llm.tokenize(text)

    def chat(self, messages, stream=False, **kw):
        self.chat_calls.append({"messages": messages, "stream": stream, **kw})

        def _gen():
            yield {"choices": [{"delta": {"content": "a"}}]}
            yield {"choices": [{"delta": {"content": "b"}}]}
            yield {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }

        return _gen()

    def close(self):
        self.closed = True


class RaisingEngine(FakeEngine):
    def chat(self, messages, stream=False, **kw):
        raise RuntimeError("bench exploded")


def test_run_bench_reports_ttft_and_throughput():
    result = bench_cmd._run_bench(FakeEngine(_plan(), "/tmp/m.gguf"), "prompt", 2)

    assert result["prompt_tok"] == 10
    assert result["gen_tok"] == 2
    assert result["token_count_is_exact"] is True
    assert result["ttft_ms"] >= 0
    assert result["total_s"] > 0


def test_run_bench_turns_engine_errors_into_a_result():
    result = bench_cmd._run_bench(RaisingEngine(_plan(), "/tmp/m.gguf"), "prompt", 2)

    assert "bench exploded" in result["error"]
    assert result["prompt_tok"] == 10


def test_run_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(bench_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))
    monkeypatch.setattr(bench_cmd, "get_model", lambda name: None)

    with pytest.raises(typer.Exit) as exc:
        bench_cmd.run(model="missing", ctx="512", max_tokens=2)

    assert exc.value.exit_code == 1


def test_run_rejects_malformed_ctx(monkeypatch):
    monkeypatch.setattr(bench_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))
    monkeypatch.setattr(
        bench_cmd,
        "get_model",
        _spec,
    )

    with pytest.raises(typer.Exit) as exc:
        bench_cmd.run(model="m", ctx="512,not-a-number", max_tokens=2)

    assert exc.value.exit_code == 1


def test_run_rejects_non_positive_ctx(monkeypatch):
    monkeypatch.setattr(bench_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))
    monkeypatch.setattr(
        bench_cmd,
        "get_model",
        _spec,
    )

    with pytest.raises(typer.Exit) as exc:
        bench_cmd.run(model="m", ctx="0", max_tokens=2)

    assert exc.value.exit_code == 1


def test_run_benchmarks_each_context_and_closes_engines(monkeypatch):
    engines = []

    class TrackingEngine(FakeEngine):
        def __init__(self, plan, path):
            super().__init__(plan, path)
            engines.append(self)

    monkeypatch.setattr(bench_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))
    monkeypatch.setattr(
        bench_cmd,
        "get_model",
        lambda name: _spec(name, {"n_layers": 4, "quant": "Q4_K_M"}),
    )
    monkeypatch.setattr(
        bench_cmd,
        "detect",
        lambda: types.SimpleNamespace(
            cpu_cores=4,
            ram_free_gb=8.0,
            gpu_name="none",
        ),
    )
    monkeypatch.setattr(bench_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(
        bench_cmd,
        "create_engine",
        lambda plan, path, backend="inprocess": TrackingEngine(plan, path),
    )
    monkeypatch.setattr(bench_cmd, "animated_loading_bar", lambda *a, **kw: None)

    bench_cmd.run(model="m", ctx="512,1024", max_tokens=2)

    assert len(engines) == 2
    assert all(e.closed for e in engines)
