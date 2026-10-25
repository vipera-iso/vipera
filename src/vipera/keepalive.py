"""Parse Ollama's ``keep_alive`` duration syntax.

Ollama accepts a number of seconds or a Go-style duration string such as
``5m``, ``1h`` or ``500ms``. Zero means "unload as soon as the request ends";
a negative value means "never unload". Keeping the parser in its own module
lets the pool and the HTTP layer share one definition without importing each
other.
"""

from __future__ import annotations

import math
import re

_DURATION = re.compile(r"^(?P<value>[+-]?\d+(?:\.\d+)?)(?P<unit>ms|s|m|h|d)?$")
_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_keep_alive(
    value: str | int | float | None, default: float | None = None
) -> float | None:
    """Return seconds, ``0`` for "unload now", or a negative value for "forever".

    ``None`` means the request omitted the field, so the caller's default is
    returned unchanged. Invalid input raises ``ValueError``; the HTTP layer turns
    that into a 400 before a model is loaded.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"invalid keep_alive: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        match = _DURATION.fullmatch(text)
        if match is None:
            raise ValueError(f"invalid keep_alive: {value!r}")
        seconds = float(match.group("value")) * _UNITS[match.group("unit") or "s"]

    if not math.isfinite(seconds):
        raise ValueError(f"invalid keep_alive: {value!r}")
    if seconds < 0:
        return -1.0
    return seconds
