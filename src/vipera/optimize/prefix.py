"""Say whether the KV cache is actually being reused across requests.

Both backends already reuse a shared prefix without help: llama-cpp-python
compares the new prompt's tokens with the ones still in the cache and only
evaluates the difference, and llama-server keeps the last prompt in its slot.
The optimization question is therefore not "add a cache" but "is the prefix
stable, and how much did it save" — a client that puts a timestamp in the
system prompt, or reorders tools between turns, silently turns every request
into a full prefill.

This module keeps that visible: it derives a stable key for the reusable part
of a request (system prompt, tool schema, history before the current turn) and
accumulates the cached-token counts the backends report.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any


def reusable_prefix(messages: list[dict], tools: list[dict] | None = None) -> dict[str, Any]:
    """The part of a request another request could share.

    That is everything up to (not including) the last user message: the system
    prompt, the tool schema and the history already answered. The turn being
    answered is by definition new, and the reply does not exist yet.
    """
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            cut = i
            break
    prefix = [dict(m) for m in messages[:cut]]
    if not prefix and not tools:
        return {}
    payload: dict[str, Any] = {"messages": prefix}
    if tools:
        payload["tools"] = tools
    return payload


def prefix_key(messages: list[dict], tools: list[dict] | None = None) -> str:
    """Stable digest of the reusable prefix; ``""`` when there is nothing to share."""
    payload = reusable_prefix(messages, tools)
    if not payload:
        return ""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


def cached_tokens(payload: dict | None) -> int:
    """Prompt tokens the backend served from its cache, 0 when it does not say.

    The OpenAI shape carries it under ``usage.prompt_tokens_details``; a raw
    llama-server reply also has ``timings.cache_n``. llama-cpp-python 0.3.x
    reports neither, so the in-process engine always counts as 0 — which is the
    honest answer to "how many did the backend tell us about".
    """
    if not isinstance(payload, dict):
        return 0
    usage = payload.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            try:
                return max(0, int(details["cached_tokens"]))
            except (TypeError, ValueError):
                return 0
    timings = payload.get("timings")
    if isinstance(timings, dict) and timings.get("cache_n") is not None:
        try:
            return max(0, int(timings["cache_n"]))
        except (TypeError, ValueError):
            return 0
    return 0


class PrefixStats:
    """Rolling counters for prefix reuse. One request at a time, but be safe."""

    requests: int
    prefix_hits: int
    prompt_tokens: int
    cached: int
    dropped: int
    truncated: int

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()
            self.requests = 0
            self.prefix_hits = 0
            self.prompt_tokens = 0
            self.cached = 0
            self.dropped = 0
            self.truncated = 0

    def record(
        self,
        key: str,
        *,
        prompt_tokens: int = 0,
        cached: int = 0,
        dropped: int = 0,
        truncated: int = 0,
    ) -> bool:
        """Count one request; return True when its prefix was seen before."""
        with self._lock:
            hit = bool(key) and key in self._seen
            if key:
                self._seen.add(key)
            self.requests += 1
            self.prefix_hits += int(hit)
            self.prompt_tokens += max(0, int(prompt_tokens))
            self.cached += max(0, int(cached))
            self.dropped += max(0, int(dropped))
            self.truncated += max(0, int(truncated))
            return hit

    def summary(self) -> dict[str, Any]:
        with self._lock:
            requests = self.requests
            prompt_tokens = self.prompt_tokens
            return {
                "requests": requests,
                # A repeat prefix is necessary but not sufficient: the backend
                # may still have evicted it, which is what cache_rate shows.
                "prefix_hits": self.prefix_hits,
                "prefix_hit_rate": round(self.prefix_hits / requests, 3) if requests else 0.0,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": self.cached,
                "cache_rate": round(self.cached / prompt_tokens, 3) if prompt_tokens else 0.0,
                "messages_dropped": self.dropped,
                "messages_truncated": self.truncated,
            }
