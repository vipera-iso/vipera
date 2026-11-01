"""What the gateway can say about the speed it just served.

`llama.cpp` knows the two numbers a user feels - how long until the first token
(prefill) and how fast the rest arrived (decode) - but it knows them inside the
child process, or not at all when the model runs in-process. The gateway times
the request at its own boundary, so what it reports is what the client saw, and
it reports it under the names `llama.cpp` already uses (`prompt_ms`, `decode_ms`,
`prompt_per_second`, `predicted_per_second`) so a tool that parses one parses the
other.

Nothing here runs on the token path: one `perf_counter()` when the request
starts, one at the first frame, one at the end.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Timings:
    """One request's time, split into prefill and decode."""

    prompt_ms: float
    decode_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def prompt_tps(self) -> float | None:
        if not self.prompt_tokens or self.prompt_ms <= 0:
            return None
        return self.prompt_tokens / (self.prompt_ms / 1000.0)

    @property
    def decode_tps(self) -> float | None:
        if not self.completion_tokens or self.decode_ms <= 0:
            return None
        return self.completion_tokens / (self.decode_ms / 1000.0)

    def to_dict(self) -> dict:
        """The shape `llama-server` prints, so existing parsers keep working."""
        return {
            "prompt_n": self.prompt_tokens,
            "prompt_ms": round(self.prompt_ms, 3),
            "prompt_per_second": round(self.prompt_tps, 3) if self.prompt_tps else None,
            "predicted_n": self.completion_tokens,
            "predicted_ms": round(self.decode_ms, 3),
            "predicted_per_second": round(self.decode_tps, 3) if self.decode_tps else None,
        }

    def line(self) -> str:
        """One line for the server log: what a user reads instead of a benchmark."""
        parts = []
        if self.prompt_tokens:
            rate = f"{self.prompt_tps:.0f} t/s" if self.prompt_tps else "?"
            parts.append(f"prefill {self.prompt_tokens} tok {self.prompt_ms:.0f} ms ({rate})")
        if self.completion_tokens:
            rate = f"{self.decode_tps:.1f} t/s" if self.decode_tps else "?"
            parts.append(f"decode {self.completion_tokens} tok {self.decode_ms:.0f} ms ({rate})")
        if not parts:
            parts.append(f"{self.prompt_ms + self.decode_ms:.0f} ms")
        return "  ".join(parts)


@dataclass
class Meter:
    """Running totals, so `/health` can answer "how fast has this been going"."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_ms: float = 0.0
    decode_ms: float = 0.0
    last: Timings | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, timings: Timings) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += timings.prompt_tokens or 0
            self.completion_tokens += timings.completion_tokens or 0
            self.prompt_ms += timings.prompt_ms
            self.decode_ms += timings.decode_ms
            self.last = timings

    def snapshot(self) -> dict:
        with self._lock:
            overall_prompt = (
                self.prompt_tokens / (self.prompt_ms / 1000.0) if self.prompt_ms > 0 else None
            )
            overall_decode = (
                self.completion_tokens / (self.decode_ms / 1000.0) if self.decode_ms > 0 else None
            )
            return {
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "prompt_tokens_per_second": round(overall_prompt, 2) if overall_prompt else None,
                "completion_tokens_per_second": round(overall_decode, 2)
                if overall_decode
                else None,
                "last": self.last.to_dict() if self.last else None,
            }


class Stopwatch:
    """Times one request: when it arrived, when the first token left, when it ended.

    A non-streaming response never shows the first token, so `first_token()` is
    simply never called and the whole duration is charged to prefill - which is
    honest for a backend that buffers its answer.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.first: float | None = None
        self.ended: float | None = None

    def first_token(self) -> None:
        if self.first is None:
            self.first = time.perf_counter()

    def finish(self) -> None:
        self.ended = time.perf_counter()

    def timings(self, usage: dict | None) -> Timings:
        """Split the elapsed time using the token counts the backend reported."""
        end = self.ended or time.perf_counter()
        split = self.first if self.first is not None else end
        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return Timings(
            prompt_ms=(split - self.started) * 1000.0,
            decode_ms=max(0.0, (end - split) * 1000.0),
            prompt_tokens=int(prompt_tokens) if prompt_tokens else None,
            completion_tokens=int(completion_tokens) if completion_tokens else None,
        )
