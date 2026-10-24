import sys
import types
from typing import ClassVar

import pytest

from vipera.engine import Engine, create_engine, load_attempts, resolve_backend
from vipera.planner import Plan


def _plan(**overrides):
    base = dict(
        gpu_layers=9,
        cpu_threads=7,
        context=65536,
        kv_cache="q8_0",
        flash_attn=True,
        n_batch=512,
        offload_kqv=True,
        use_mlock=False,
        note="Hybrid",
    )
    base.update(overrides)
    return Plan(**base)


class FakeLlama:
    """Stand-in for llama_cpp.Llama that records loads and can fail on cue."""

    calls: ClassVar[list[dict]] = []
    chat_calls: ClassVar[list[dict]] = []
    tokenize_calls: ClassVar[list[bytes]] = []
    embed_calls: ClassVar[list[list[str]]] = []
    per_token: ClassVar[bool] = False
    fail_if = staticmethod(lambda kwargs: False)

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        if type(self).fail_if(kwargs):
            raise RuntimeError("simulated load failure")

    def create_chat_completion(self, **kwargs):
        type(self).chat_calls.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def tokenize(self, data):
        type(self).tokenize_calls.append(data)
        return [1, 2, 3]

    def create_embedding(self, texts):
        type(self).embed_calls.append(list(texts))
        if type(self).per_token:
            # What a build that ignores pooling_type returns: one vector per
            # token.
            return {"data": [{"embedding": [[1.0, 2.0], [3.0, 5.0]]} for _ in texts]}
        return {"data": [{"embedding": [float(len(t))]} for t in texts]}


@pytest.fixture(autouse=True)
def fake_llama(monkeypatch):
    FakeLlama.calls = []
    FakeLlama.chat_calls = []
    FakeLlama.tokenize_calls = []
    FakeLlama.embed_calls = []
    FakeLlama.per_token = False
    FakeLlama.fail_if = staticmethod(lambda kwargs: False)
    module = types.SimpleNamespace(Llama=FakeLlama, LLAMA_POOLING_TYPE_MEAN=1)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return FakeLlama


def test_insufficient_ram_is_refused_before_loading(fake_llama):
    # OOM is a SIGKILL, so the staged fallback cannot catch it; fail cleanly.
    plan = _plan(ram_estimate_gb=9999.0)
    with pytest.raises(RuntimeError, match="Not enough RAM"):
        Engine(plan, "model.gguf")
    assert fake_llama.calls == []


def test_gpu_fallback_walks_the_offload_down_before_cpu(fake_llama):
    # 9 layers planned, only 6 fit: the ladder is 9, 8, 7, then a quarter (2).
    # Measured on the reference box, asking for 15 layers loads 13 instead of
    # CPU-only, and the last rung is deliberately small: a failed attempt keeps
    # its VRAM, so after three failures even a 10-layer rung failed.
    fake_llama.fail_if = staticmethod(lambda kw: kw["n_gpu_layers"] > 6)

    engine = Engine(_plan(note="Hybrid: 9/40 layers"), "model.gguf")

    assert [c["n_gpu_layers"] for c in fake_llama.calls] == [9, 8, 7, 2]
    assert engine.plan.gpu_layers == 2
    # The note follows the split that actually loaded, not the planned one.
    assert engine.plan.note == "Hybrid: 2/40 layers"


def test_reduced_offload_does_not_still_claim_full_gpu(fake_llama):
    # A big card can plan a full offload and still fail: the note must follow the
    # split that loaded instead of saying "Full GPU" while 1 layer sits on CPU.
    fake_llama.fail_if = staticmethod(lambda kw: kw["n_gpu_layers"] >= 32)

    engine = Engine(_plan(gpu_layers=32, note="Full GPU"), "model.gguf")

    assert engine.plan.gpu_layers == 31
    assert engine.plan.note == "Hybrid: 31/all layers"


def test_cpu_fallback_keeps_context_kv_and_flash_attention(fake_llama):
    fake_llama.fail_if = staticmethod(lambda kw: kw["n_gpu_layers"] > 0)

    engine = Engine(_plan(), "model.gguf")

    assert [c["n_gpu_layers"] for c in fake_llama.calls] == [9, 8, 7, 2, 0]
    gpu, cpu = fake_llama.calls[0], fake_llama.calls[-1]

    assert gpu["n_gpu_layers"] == 9
    assert cpu["n_gpu_layers"] == 0
    # The whole point: do not silently drop to 2048/f16.
    assert cpu["n_ctx"] == 65536
    assert cpu["type_k"] == 8  # q8_0
    assert cpu["type_v"] == 8
    assert cpu["flash_attn"] is True
    assert cpu["offload_kqv"] is False
    assert engine.plan.context == 65536
    assert engine.plan.kv_cache == "q8_0"


def test_cpu_fallback_reduces_context_only_as_second_step(fake_llama, capsys):
    fake_llama.fail_if = staticmethod(lambda kw: kw["n_ctx"] > 2048)

    engine = Engine(_plan(), "model.gguf")

    # Every GPU rung and the full-context CPU rung keep 65536; only the second
    # CPU stage (after the GPU ladder) drops it.
    assert [c["n_ctx"] for c in fake_llama.calls] == [65536, 65536, 65536, 65536, 65536, 2048]
    # KV quantization survives the context reduction.
    assert fake_llama.calls[-1]["type_k"] == 8
    assert engine.plan.context == 2048
    assert "will NOT be met" in capsys.readouterr().out


def test_final_fallback_uses_f16_without_flash_attention(fake_llama):
    fake_llama.fail_if = staticmethod(lambda kw: kw["type_k"] == 8)

    engine = Engine(_plan(), "model.gguf")

    final = fake_llama.calls[-1]
    assert final["n_ctx"] == 2048
    assert final["type_k"] == 1  # f16
    assert final["flash_attn"] is False
    assert engine.plan.kv_cache == "f16"


def test_chat_forwards_openai_sampling_and_tool_params(fake_llama):
    engine = Engine(_plan(), "model.gguf")
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]

    engine.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
        temperature=0.2,
        max_tokens=32,
        top_p=0.8,
        top_k=10,
        min_p=0.01,
        repeat_penalty=1.2,
        presence_penalty=0.1,
        frequency_penalty=-0.1,
        stop=["\n\n"],
        seed=7,
        logit_bias={"42": -2.0},
        logprobs=True,
        top_logprobs=3,
        response_format={"type": "json_object"},
        tools=tools,
        tool_choice="auto",
    )

    kwargs = fake_llama.chat_calls[-1]
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["stream"] is True
    assert kwargs["temperature"] == pytest.approx(0.2)
    assert kwargs["max_tokens"] == 32
    assert kwargs["top_p"] == pytest.approx(0.8)
    assert kwargs["top_k"] == 10
    assert kwargs["min_p"] == pytest.approx(0.01)
    assert kwargs["repeat_penalty"] == pytest.approx(1.2)
    assert kwargs["presence_penalty"] == pytest.approx(0.1)
    assert kwargs["frequency_penalty"] == pytest.approx(-0.1)
    assert kwargs["stop"] == ["\n\n"]
    assert kwargs["seed"] == 7
    assert kwargs["logit_bias"] == {"42": -2.0}
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 3
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


def test_chat_drops_unknown_and_unset_params(fake_llama):
    engine = Engine(_plan(), "model.gguf")

    engine.chat(
        [{"role": "user", "content": "hi"}],
        user="someone",  # not a llama.cpp chat parameter
        stream_options={"include_usage": True},
        top_k=None,  # unset optional knob
        seed=None,
    )

    kwargs = fake_llama.chat_calls[-1]
    assert "user" not in kwargs
    assert "stream_options" not in kwargs
    assert "top_k" not in kwargs
    assert "seed" not in kwargs
    # Documented defaults still apply when the caller sends nothing.
    assert kwargs["temperature"] == pytest.approx(0.7)
    assert kwargs["max_tokens"] == 2048
    assert kwargs["top_p"] == pytest.approx(0.95)
    assert kwargs["repeat_penalty"] == pytest.approx(1.1)


def test_tokenize_goes_through_the_loaded_model(fake_llama):
    engine = Engine(_plan(), "model.gguf")

    assert engine.tokenize("hello") == [1, 2, 3]
    assert fake_llama.tokenize_calls[-1] == b"hello"

    # Already-encoded bytes pass through untouched (bench used to call
    # engine.llm.tokenize(prompt.encode()) directly).
    engine.tokenize(b"raw bytes")
    assert fake_llama.tokenize_calls[-1] == b"raw bytes"


def test_create_engine_defaults_to_the_in_process_engine(fake_llama):
    engine = create_engine(_plan(), "model.gguf")

    assert isinstance(engine, Engine)
    assert engine.plan.context == 65536


def test_engine_does_not_ask_llama_for_a_draft_model(fake_llama):
    # Prompt-lookup drafting inside llama-cpp-python measured -29% decode
    # (4.27 t/s against 6.05), so the in-process engine leaves it alone: the
    # server backend passes llama.cpp's own --spec-type instead.
    Engine(_plan(), "model.gguf")

    assert "draft_model" not in fake_llama.calls[0]


def test_create_engine_selects_the_server_backend(monkeypatch):
    seen = {}

    class StubServer:
        def __init__(self, plan, path):
            seen["plan"] = plan
            seen["path"] = path

    monkeypatch.setattr("vipera.backends.server.ServerEngine", StubServer)

    engine = create_engine(_plan(), "model.gguf", "server")

    assert isinstance(engine, StubServer)
    assert seen["path"] == "model.gguf"


def test_resolve_backend_upgrades_an_moe_plan(monkeypatch, capsys):
    monkeypatch.setattr(
        "vipera.backends.registry.find_llama_tool", lambda *a, **kw: "/bin/llama-server"
    )
    monkeypatch.setattr("vipera.backends.server.supports_n_cpu_moe", lambda binary: True)
    monkeypatch.setattr(
        "vipera.backends.server.probe_devices",
        lambda binary: "Available devices:\n  CUDA0: NVIDIA GeForce GTX 1650\n",
    )

    backend, plan = resolve_backend(
        _plan(moe_cpu_layers=12, n_layers=24, n_ubatch=896), "inprocess"
    )

    assert backend == "server"
    assert plan.requires_server is True
    assert plan.moe_cpu_layers == 12
    assert "expert placement" in capsys.readouterr().out


def test_resolve_backend_refuses_a_cpu_only_server(monkeypatch, capsys):
    # A build can know -ncmoe and still be CPU-only (or miss its CUDA runtime
    # libraries). It must not displace a faster in-process GPU split.
    monkeypatch.setattr(
        "vipera.backends.registry.find_llama_tool", lambda *a, **kw: "/bin/llama-server"
    )
    monkeypatch.setattr("vipera.backends.server.supports_n_cpu_moe", lambda binary: True)
    monkeypatch.setattr(
        "vipera.backends.server.probe_devices",
        lambda binary: "Available devices:\n  (none)\n",
    )

    backend, plan = resolve_backend(
        _plan(gpu_layers=14, moe_cpu_layers=12, n_layers=24, n_ubatch=896), "inprocess"
    )

    out = capsys.readouterr().out
    assert backend == "inprocess"
    assert plan.requires_server is False
    assert plan.moe_cpu_layers == -1
    assert plan.n_ubatch == 0
    assert plan.gpu_layers == 14
    assert "CUDA0" in out
    assert "falling back" in out
    assert "LD_LIBRARY_PATH" in out


def test_resolve_backend_treats_a_failed_probe_as_no_cuda(monkeypatch, capsys):
    monkeypatch.setattr(
        "vipera.backends.registry.find_llama_tool", lambda *a, **kw: "/bin/llama-server"
    )
    monkeypatch.setattr("vipera.backends.server.supports_n_cpu_moe", lambda binary: True)
    monkeypatch.setattr("vipera.backends.server.probe_devices", lambda binary: None)

    backend, plan = resolve_backend(
        _plan(gpu_layers=14, moe_cpu_layers=12, n_layers=24, n_ubatch=896), "inprocess"
    )

    out = capsys.readouterr().out
    assert backend == "inprocess"
    assert plan.moe_cpu_layers == -1
    assert "probe failed" in out


def test_resolve_backend_falls_back_when_ncmoe_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        "vipera.backends.registry.find_llama_tool", lambda *a, **kw: "/bin/llama-server"
    )
    monkeypatch.setattr("vipera.backends.server.supports_n_cpu_moe", lambda binary: False)

    backend, plan = resolve_backend(
        _plan(gpu_layers=14, moe_cpu_layers=12, n_layers=24, n_ubatch=896), "inprocess"
    )

    assert backend == "inprocess"
    # The fallback must not keep claiming the MoE optimization.
    assert plan.requires_server is False
    assert plan.moe_cpu_layers == -1
    assert plan.n_ubatch == 0
    assert plan.gpu_layers == 14
    assert "falling back" in capsys.readouterr().out


def test_embeddings_are_off_unless_the_config_enables_them(fake_llama, monkeypatch):
    from vipera.backends import inprocess as inprocess_mod
    from vipera.errors import BackendError

    monkeypatch.setattr(inprocess_mod.config, "load_config", lambda: {"embeddings": False})
    engine = Engine(_plan(), "/tmp/m.gguf")

    assert engine.supports_embeddings is False
    assert fake_llama.calls[-1]["embedding"] is False
    with pytest.raises(BackendError):
        engine.embed(["hi"])


def test_embeddings_load_with_mean_pooling(fake_llama, monkeypatch):
    from vipera.backends import inprocess as inprocess_mod

    monkeypatch.setattr(inprocess_mod.config, "load_config", lambda: {"embeddings": True})
    engine = Engine(_plan(), "/tmp/m.gguf")

    kwargs = fake_llama.calls[-1]
    assert kwargs["embedding"] is True
    # Without a pooling type llama.cpp returns one vector per token; the
    # fixture's LLAMA_POOLING_TYPE_MEAN is 1.
    assert kwargs["pooling_type"] == 1
    assert engine.embed(["hi", "there"]) == [[2.0], [5.0]]


def test_per_token_vectors_are_mean_pooled(fake_llama, monkeypatch):
    from vipera.backends import inprocess as inprocess_mod

    monkeypatch.setattr(inprocess_mod.config, "load_config", lambda: {"embeddings": True})
    fake_llama.per_token = True
    engine = Engine(_plan(), "/tmp/m.gguf")

    assert engine.embed(["hi"]) == [[2.0, 3.5]]


def test_moe_load_attempts_step_experts_down_then_go_cpu():
    plan = _plan(gpu_layers=14, moe_cpu_layers=12, n_layers=24, n_ubatch=896, note="MoE")

    attempts = load_attempts(plan)
    moe = [candidate for candidate, _ in attempts if candidate.requires_server]
    cpu = [candidate for candidate, _ in attempts if candidate.gpu_layers == 0]

    assert [candidate.moe_cpu_layers for candidate in moe] == [12, 13, 14, 24]
    assert cpu
    assert all(candidate.moe_cpu_layers == -1 for candidate in cpu)
    assert all(candidate.n_ubatch == 0 for candidate in cpu)
