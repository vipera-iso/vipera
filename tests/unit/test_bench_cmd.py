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


class StreamEngine(FakeEngine):
    """Yields an explicit chunk list, so a stream shape can be asserted."""

    def __init__(self, plan, path, chunks):
        super().__init__(plan, path)
        self.chunks = chunks

    def chat(self, messages, stream=False, **kw):
        self.chat_calls.append({"messages": messages, "stream": stream, **kw})
        return iter(self.chunks)


class Clock:
    """Deterministic perf_counter: one value per call, then the last one."""

    def __init__(self, times):
        self.times = list(times)
        self.calls = 0

    def __call__(self):
        value = self.times[min(self.calls, len(self.times) - 1)]
        self.calls += 1
        return value


def _reasoning(text):
    return {"choices": [{"delta": {"reasoning_content": text}}]}


def _content(text):
    return {"choices": [{"delta": {"content": text}}]}


def _usage(prompt_tokens, completion_tokens=None):
    usage = {"prompt_tokens": prompt_tokens}
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    return {"choices": [], "usage": usage}


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


def test_run_bench_measures_ttft_from_the_first_reasoning_chunk(monkeypatch):
    # A reasoning model emits reasoning_content before any content; that is the
    # first token, and ignoring it made ttft equal the whole generation.
    engine = StreamEngine(
        _plan(),
        "/tmp/m.gguf",
        [
            _reasoning("think"),
            _reasoning("ing"),
            _content("answer"),
            _usage(55, 41),
        ],
    )
    monkeypatch.setattr(bench_cmd.time, "perf_counter", Clock([0.0, 0.5, 2.0]))

    result = bench_cmd._run_bench(engine, "prompt", 64)

    assert result["ttft_ms"] == pytest.approx(500.0)
    assert result["prompt_tok"] == 55
    assert result["prompt_tps"] == pytest.approx(110.0)
    assert result["gen_tok"] == 41
    assert result["token_count_is_exact"] is True
    assert result["gen_tps"] == pytest.approx(40 / 1.5)


def test_run_bench_handles_a_reasoning_only_stream(monkeypatch):
    engine = StreamEngine(
        _plan(),
        "/tmp/m.gguf",
        [_reasoning("hmm"), _reasoning("more"), _usage(55, 30)],
    )
    monkeypatch.setattr(bench_cmd.time, "perf_counter", Clock([0.0, 0.4, 1.4]))

    result = bench_cmd._run_bench(engine, "prompt", 64)

    assert result["ttft_ms"] == pytest.approx(400.0)
    assert result["total_s"] == pytest.approx(1.4)
    assert result["gen_tps"] == pytest.approx(29 / 1.0)


def test_run_bench_counts_reasoning_chunks_when_usage_is_missing(monkeypatch):
    engine = StreamEngine(
        _plan(),
        "/tmp/m.gguf",
        [_reasoning("a"), _reasoning("b"), _content("c")],
    )
    monkeypatch.setattr(bench_cmd.time, "perf_counter", Clock([0.0, 0.5, 2.0]))

    result = bench_cmd._run_bench(engine, "prompt", 64)

    assert result["token_count_is_exact"] is False
    assert result["gen_tok"] == 3
    assert result["prompt_tok"] == 10  # tokenize fallback, no server usage


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


def test_bench_uses_the_configured_planner_profile(monkeypatch):
    """`profile` in config must reach the planner here too, not just in serve."""
    seen = {}
    hw = types.SimpleNamespace(
        cpu_cores=4, ram_free_gb=8.0, gpu_name="none", ram_total_gb=16.0, vram_total_gb=0.0
    )
    monkeypatch.setattr(bench_cmd, "load_config", lambda: {**DEFAULT_CONFIG, "profile": "cpu"})
    monkeypatch.setattr(bench_cmd, "get_model", _spec)
    monkeypatch.setattr(bench_cmd, "detect", lambda: hw)

    def fake_plan(*args, **kwargs):
        seen.update(kwargs)
        return _plan()

    monkeypatch.setattr(bench_cmd, "plan", fake_plan)
    monkeypatch.setattr(bench_cmd, "create_engine", lambda p, path, backend: FakeEngine(p, path))
    monkeypatch.setattr(bench_cmd, "animated_loading_bar", lambda *a, **kw: None)

    bench_cmd.run(model="m", ctx="512", max_tokens=2)

    assert seen["profile"] == "cpu"


def test_bench_records_the_split_that_ran(monkeypatch, tmp_path):
    from vipera.optimize.tuning import TuningStore

    store = TuningStore(tmp_path / "tuning.json")
    hw = types.SimpleNamespace(
        cpu_cores=4, ram_free_gb=8.0, gpu_name="GTX 1650", ram_total_gb=32.0, vram_total_gb=4.0
    )
    monkeypatch.setattr(bench_cmd, "TuningStore", lambda: store)
    monkeypatch.setattr(bench_cmd, "load_config", lambda: dict(DEFAULT_CONFIG))
    monkeypatch.setattr(bench_cmd, "get_model", _spec)
    monkeypatch.setattr(bench_cmd, "detect", lambda: hw)
    monkeypatch.setattr(bench_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(bench_cmd, "create_engine", lambda p, path, backend: FakeEngine(p, path))
    monkeypatch.setattr(bench_cmd, "animated_loading_bar", lambda *a, **kw: None)

    bench_cmd.run(model="m", ctx="512", max_tokens=2)

    profile = store.get("m", hw)
    assert profile is not None
    assert profile.gpu_layers == 2
    assert profile.ctx == 512
    assert profile.gen_tps > 0


def test_bench_stores_nothing_when_tuning_is_off(monkeypatch, tmp_path):
    from vipera.optimize.tuning import TuningStore

    store = TuningStore(tmp_path / "tuning.json")
    hw = types.SimpleNamespace(
        cpu_cores=4, ram_free_gb=8.0, gpu_name="GTX 1650", ram_total_gb=32.0, vram_total_gb=4.0
    )
    monkeypatch.setattr(bench_cmd, "TuningStore", lambda: store)
    monkeypatch.setattr(bench_cmd, "load_config", lambda: {**DEFAULT_CONFIG, "tuning": False})
    monkeypatch.setattr(bench_cmd, "get_model", _spec)
    monkeypatch.setattr(bench_cmd, "detect", lambda: hw)
    monkeypatch.setattr(bench_cmd, "plan", lambda *a, **kw: _plan())
    monkeypatch.setattr(bench_cmd, "create_engine", lambda p, path, backend: FakeEngine(p, path))
    monkeypatch.setattr(bench_cmd, "animated_loading_bar", lambda *a, **kw: None)

    bench_cmd.run(model="m", ctx="512", max_tokens=2)

    assert not (tmp_path / "tuning.json").exists()
