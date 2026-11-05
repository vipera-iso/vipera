"""Client for a gateway that is already running (`vipera serve`).

`vipera chat` used to always load its own copy of the model. On a box with one
card that copy competes with the gateway for VRAM, and on a model larger than
RAM it competes for the page cache too: 62 GB read a second time so that a REPL
can ask one question. If the gateway in the config is up and lists the alias,
chat talks to it instead.

Deliberately not a `ChatEngine`: this client never loaded the model, so it has
no `Plan` of its own to report, and inventing one would put a second, fake
placement back in front of the user.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from typing import Any

from vipera.backends.protocol import CHAT_PARAMS, DEFAULT_SAMPLING
from vipera.backends.server import _request_timeout

#: The gateway is local; a probe that has not answered by now is not there.
_PROBE_TIMEOUT_S = 2.0


def base_url(host: str, port: int) -> str:
    """The URL to reach a gateway bound to ``host``.

    A wildcard bind (`0.0.0.0`, `::`) is not a destination: connecting to it
    works on Linux only by accident of the routing table.
    """
    if host in ("", "0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def served_models(
    base_url: str, *, api_key: str | None = None, timeout: float = _PROBE_TIMEOUT_S
) -> list[str] | None:
    """Model ids the gateway lists, or None when nothing answers there.

    None and an empty list are different answers: the first means \"no gateway\",
    the second \"a gateway that serves nothing\".
    """
    # Local import: urllib drags in ssl and email, and `vipera serve` imports
    # this module at startup for a probe it only runs when attaching.
    import urllib.request

    request = urllib.request.Request(base_url.rstrip("/") + "/v1/models", headers=_headers(api_key))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (OSError, TimeoutError, ValueError):
        # URLError/HTTPError are OSErrors; a body that is not JSON is a different
        # program on that port, which is also not a gateway to attach to.
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]


class GatewayEngine:
    """Chat through a running gateway, as one more OpenAI client."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key

    def _post(self, path: str, body: dict, *, timeout: float):
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **_headers(self._api_key)},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"gateway {path} returned {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"gateway is unreachable: {exc.reason}") from None
        except TimeoutError:
            raise RuntimeError(f"gateway did not answer on {path} within {timeout:.0f}s") from None

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any):
        params = {k: v for k, v in kw.items() if k in CHAT_PARAMS and v is not None}
        for key, default in DEFAULT_SAMPLING.items():
            params.setdefault(key, default)
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        body.update(params)
        timeout = _request_timeout(params.get("max_tokens"))
        response = self._post("/v1/chat/completions", body, timeout=timeout)
        if not stream:
            with response:
                return json.loads(response.read())
        return self._frames(response, timeout=timeout)

    @staticmethod
    def _frames(response, *, timeout: float) -> Iterator[dict]:
        try:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
        except TimeoutError:
            raise RuntimeError(f"gateway did not send a token within {timeout:.0f}s") from None
        finally:
            with contextlib.suppress(OSError):
                response.close()

    def close(self) -> None:
        """Nothing to stop: the gateway owns the model, not this client."""


def attach(base_url: str, model: str, *, api_key: str | None = None) -> GatewayEngine | None:
    """A client for the gateway at ``base_url`` when it serves ``model``."""
    ids = served_models(base_url, api_key=api_key)
    if not ids or model not in ids:
        return None
    return GatewayEngine(base_url, model, api_key=api_key)


__all__ = ["GatewayEngine", "attach", "base_url", "served_models"]
