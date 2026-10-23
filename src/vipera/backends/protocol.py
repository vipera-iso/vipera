"""The contract every backend implements and the parameters it accepts.

Both engines take the same OpenAI-shaped kwargs. `CHAT_PARAMS` is the allowlist
that reaches llama.cpp: anything else is dropped, so a typo in an HTTP request
cannot turn into a `TypeError` from the binding.
"""

from __future__ import annotations

from typing import Any, Protocol

from vipera.planner import Plan


class ChatEngine(Protocol):
    """What the commands need from either backend.

    The in-process engine and the llama-server child both satisfy this; callers
    must not reach for `.llm`, which only the in-process one has.
    """

    plan: Plan

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any): ...

    def tokenize(self, text: str | bytes) -> list[int]: ...

    def close(self) -> None: ...


CHAT_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "max_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "tfs_z",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "grammar",
        "response_format",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
    }
)

DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
}
