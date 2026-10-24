"""The gateway side of the optimization layer.

Two jobs, both inside the request path:

* before a call, shape the messages into the context the model was loaded with
  (:mod:`vipera.optimize.shaping`);
* after a call, account for the reusable prefix and the tokens the backend
  served from its cache (:mod:`vipera.optimize.prefix`), so `/health` can answer
  "is the agent loop actually reusing anything".

The layer never changes sampling, routing or the reply shape: a request that
fits is passed through untouched, and the accounting is read-only.
"""

from __future__ import annotations

from typing import Any

from vipera.optimize.prefix import PrefixStats, cached_tokens, prefix_key
from vipera.optimize.shaping import Shaping, shape


class OptimizeLayer:
    """Per-gateway optimization state (one instance per served app)."""

    def __init__(self, *, shaping: bool = True) -> None:
        self.shaping_enabled = shaping
        self.stats = PrefixStats()

    @classmethod
    def from_config(cls) -> OptimizeLayer:
        """Build from the config file; a missing key keeps shaping on."""
        from vipera.config import load_config

        return cls(shaping=bool(load_config().get("shaping", True)))

    def prepare(
        self,
        engine: Any,
        messages: list[dict],
        *,
        max_tokens: int = 0,
        tools: list[dict] | None = None,
    ) -> Shaping:
        """Trim ``messages`` to the engine's context; a no-op when it fits."""
        if not self.shaping_enabled:
            return Shaping(messages=list(messages))
        context = int(getattr(getattr(engine, "plan", None), "context", 0) or 0)
        return shape(messages, budget_tokens=context, max_tokens=max_tokens)

    def note(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        payload: dict | None = None,
        shaping: Shaping | None = None,
    ) -> bool:
        """Account for one request; return True when its prefix was seen before.

        ``payload`` is the backend's reply (or a ``{"usage": ...}`` wrapper for a
        stream). When it carries no exact prompt count the shaping estimate is
        used, which is close enough for the ratio this feeds.
        """
        prompt_tokens = 0
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            try:
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
            except (TypeError, ValueError):
                prompt_tokens = 0
        if prompt_tokens <= 0 and shaping is not None:
            prompt_tokens = shaping.estimated_after
        return self.stats.record(
            prefix_key(messages, tools),
            prompt_tokens=prompt_tokens,
            cached=cached_tokens(payload),
            dropped=shaping.dropped if shaping is not None else 0,
            truncated=shaping.truncated if shaping is not None else 0,
        )
