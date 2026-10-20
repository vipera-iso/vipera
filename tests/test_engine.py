import sys
import types
from typing import ClassVar

import pytest

from vipera.engine import Engine
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
    fail_if = staticmethod(lambda kwargs: False)

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        if type(self).fail_if(kwargs):
            raise RuntimeError("simulated load failure")

    def create_chat_completion(self, **kwargs):
        type(self).chat_calls.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


@pytest.fixture(autouse=True)
def fake_llama(monkeypatch):
    FakeLlama.calls = []
    FakeLlama.chat_calls = []
    FakeLlama.fail_if = staticmethod(lambda kwargs: False)
    module = types.SimpleNamespace(Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return FakeLlama


def test_insufficient_ram_is_refused_before_loading(fake_llama):
    # OOM is a SIGKILL, so the staged fallback cannot catch it; fail cleanly.
    plan = _plan(ram_estimate_gb=9999.0)
    with pytest.raises(RuntimeError, match="Not enough RAM"):
        Engine(plan, "model.gguf")
    assert fake_llama.calls == []


def test_cpu_fallback_keeps_context_kv_and_flash_attention(fake_llama):
    fake_llama.fail_if = staticmethod(lambda kw: kw["n_gpu_layers"] > 0)

    engine = Engine(_plan(), "model.gguf")

    assert len(fake_llama.calls) == 2
    gpu, cpu = fake_llama.calls

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

    assert len(fake_llama.calls) == 3
    assert fake_llama.calls[0]["n_ctx"] == 65536
    assert fake_llama.calls[1]["n_ctx"] == 65536
    assert fake_llama.calls[2]["n_ctx"] == 2048
    # KV quantization survives the context reduction.
    assert fake_llama.calls[2]["type_k"] == 8
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
    tools = [{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }]

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
        user="someone",              # not a llama.cpp chat parameter
        stream_options={"include_usage": True},
        top_k=None,                  # unset optional knob
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
