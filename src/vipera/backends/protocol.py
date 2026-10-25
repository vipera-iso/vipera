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

    def health(self) -> str:
        """One word the gateway's ``/health`` can report.

        ``ok`` when the engine is serving; otherwise why not (``starting``,
        ``exited``, ``unreachable``). Probing must not start or restart it.
        """
        ...

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

# Filled in when a caller sends nothing. `repeat_penalty` is deliberately not
# here: llama.cpp's own default is 1.0, an OpenAI client never sends the
# parameter, and on the reference box charging it on every request cost ~10%
# of decode (sampler work per token) for a setting nobody asked for. A caller
# that wants it passes it explicitly.
DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
}
