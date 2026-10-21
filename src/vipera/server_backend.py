"""llama-server backend.

Runs the `llama-server` binary as a child process and speaks the OpenAI HTTP
API to it. The public surface is the same as `vipera.engine.Engine`:
`.plan`, `.chat()`, `.tokenize()`, `.close()`.

Why this exists: the Python binding bundles its own llama.cpp and exposes only
part of it. A locally built `llama-server` can be newer (better CPU kernels,
new flags) and is the only way to use builds vipera does not ship. Selecting it
is opt-in through the `backend` config key; `inprocess` stays the default until
this path has been measured on real hardware.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from vipera import config
from vipera.engine import _CHAT_PARAMS, _DEFAULT_SAMPLING, check_host_ram
from vipera.optimizer import OptimizerError, find_llama_tool
from vipera.planner import Plan

# Loading a 14B from a cold page cache on CPU takes minutes; the HTTP server
# binds first and answers 503 until the model is ready, so poll for a while.
_START_TIMEOUT_S = 900
_HEALTH_INTERVAL_S = 0.5
_STOP_TIMEOUT_S = 10.0

# Printed by llama.cpp builds without a GPU backend. The child keeps running
# but ignores --gpu-layers, so the plan must be corrected instead of reporting
# layers that are not on the GPU.
_GPU_IGNORED_MARKER = "--gpu-layers option will be ignored"

# Chat parameters llama-server understands. It ignores unknown JSON fields, but
# sending a curated set keeps a typo from becoming a 400 in the child.
_SERVER_CHAT_PARAMS = _CHAT_PARAMS - {"functions", "function_call", "grammar"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _supports_load_mode(binary: str) -> bool:
    """Whether this llama-server knows `--load-mode`.

    There is no `--mlock` on the server (that is a llama-cli flag); the mlock
    equivalent is a `--load-mode` value, and builds that predate it accept the
    flag only to exit with "invalid argument" after the weights are mapped.
    One `--help` run per spawn is cheaper than a failed child, and the caller
    may skip the probe entirely by passing the answer in.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--load-mode" in f"{probe.stdout}{probe.stderr}"


def server_command(
    binary: str,
    model_path: str,
    plan: Plan,
    port: int,
    *,
    load_mode_supported: bool | None = None,
) -> list[str]:
    """Translate a Plan into llama-server flags.

    `--jinja` is required for tool calling and for models whose chat template
    is not one of the hard-coded ones; it is harmless for plain chat.

    `load_mode_supported` skips the `--help` probe; `None` probes the binary.
    """
    command = [
        binary,
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(plan.context),
        "-ngl",
        str(plan.gpu_layers),
        "-t",
        str(plan.cpu_threads),
        "-b",
        str(plan.n_batch),
        "-ub",
        str(min(plan.n_batch, 512)),
        "-fa",
        "on" if plan.flash_attn else "off",
        "--jinja",
    ]
    if plan.kv_cache != "f16":
        command.extend(["-ctk", plan.kv_cache, "-ctv", plan.kv_cache])
    if plan.use_mlock:
        supported = (
            _supports_load_mode(binary) if load_mode_supported is None else load_mode_supported
        )
        if supported:
            command.extend(["-lm", "mlock"])
    return command


class ServerEngine:
    """Engine-compatible client for a `llama-server` child process."""

    def __init__(self, plan: Plan, model_path: str):
        check_host_ram(plan)

        self.plan = plan
        self._model_path = model_path
        self._closed = False
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._port: int | None = None
        self._log_path: Path | None = None

        try:
            self._binary = find_llama_tool("server")
        except OptimizerError as exc:
            raise RuntimeError(
                f"{exc} The 'server' backend needs the llama-server binary; "
                "the Python binding does not ship one."
            ) from exc

        try:
            self._start(plan)
        except RuntimeError as exc:
            if plan.gpu_layers == 0:
                raise
            # Same honest fallback as the in-process engine: a plan the planner
            # believed in can still fail to allocate, and CPU-only is a real
            # (slow) answer instead of a crash.
            print(f"\n[!] Warning: llama-server failed to start ({exc}). Falling back to CPU...")
            cpu_plan = replace(
                plan,
                gpu_layers=0,
                offload_kqv=False,
                n_batch=min(plan.n_batch, 128),
            )
            try:
                self._start(cpu_plan)
                print(f"[!] Warning: serving CPU-only with ctx={cpu_plan.context}.")
            except RuntimeError as cpu_exc:
                raise RuntimeError(
                    f"llama-server failed to start on GPU and CPU: {cpu_exc}"
                ) from cpu_exc

        atexit.register(self.close)

    # -- lifecycle ---------------------------------------------------------

    def _start(self, plan: Plan) -> None:
        port = _free_port()
        log_path = config.LOGS_DIR / f"llama-server-{port}.log"
        command = server_command(self._binary, self._model_path, plan, port)
        # The handle is handed to Popen and must outlive this function.
        handle = open(log_path, "wb")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"could not run {self._binary}: {exc}") from exc

        self._proc, self._log, self._port, self._log_path = proc, handle, port, log_path
        try:
            self._wait_ready(proc, port)
        except RuntimeError:
            self._stop_process()
            raise
        self.plan = self._adopt_placement(plan)

    def _wait_ready(self, proc: subprocess.Popen, port: int) -> None:
        deadline = time.monotonic() + _START_TIMEOUT_S
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if (rc := proc.poll()) is not None:
                raise RuntimeError(f"llama-server exited with code {rc}{self._tail()}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(_HEALTH_INTERVAL_S)
        raise RuntimeError(
            f"llama-server did not become healthy within {_START_TIMEOUT_S}s{self._tail()}"
        )

    def _log_text(self, limit: int = 200_000) -> str:
        if self._log_path is None:
            return ""
        with contextlib.suppress(OSError):
            return self._log_path.read_text(errors="replace")[-limit:]
        return ""

    def _tail(self) -> str:
        text = self._log_text(1500)
        if text.strip():
            return f"\n--- llama-server log ({self._log_path}) ---\n{text}"
        return ""

    def _adopt_placement(self, plan: Plan) -> Plan:
        """Report what the child actually did, not what was asked.

        A CPU-only llama.cpp build starts normally and silently drops
        --gpu-layers, which would otherwise make the banner and `bench` claim
        layers that never left system RAM.
        """
        if plan.gpu_layers and _GPU_IGNORED_MARKER in self._log_text():
            print(
                "[!] Warning: llama-server was built without GPU support and "
                f"ignored --gpu-layers {plan.gpu_layers}; serving CPU-only."
            )
            return replace(
                plan,
                gpu_layers=0,
                offload_kqv=False,
                note="CPU ONLY (llama-server ignored --gpu-layers)",
            )
        return plan

    def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_process()

    @property
    def log_path(self) -> Path | None:
        """Where the child's stdout/stderr went; useful in bug reports."""
        return self._log_path

    # -- inference ---------------------------------------------------------

    @property
    def _base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("llama-server is not running")
        return f"http://127.0.0.1:{self._port}"

    def _post(self, path: str, body: dict):
        request = urllib.request.Request(
            self._base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=None)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"llama-server {path} returned {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llama-server is unreachable: {exc.reason}") from None

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any):
        params = {k: v for k, v in kw.items() if k in _SERVER_CHAT_PARAMS and v is not None}
        for key, default in _DEFAULT_SAMPLING.items():
            params.setdefault(key, default)
        body: dict[str, Any] = {"messages": messages, "stream": stream, **params}
        # Not part of the in-process engine's surface, but llama-server uses it
        # to put the exact token usage in the final SSE chunk; bench relies on
        # that for an exact generation count.
        if kw.get("stream_options") is not None:
            body["stream_options"] = kw["stream_options"]

        if not stream:
            with self._post("/v1/chat/completions", body) as resp:
                return json.loads(resp.read())

        # A streaming generator keeps the HTTP response open across yields, so
        # the caller can consume it exactly like llama.cpp's own generator.
        response = self._post("/v1/chat/completions", body)
        return self._iter_sse(response)

    @staticmethod
    def _iter_sse(response) -> Iterator[dict]:
        try:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                with contextlib.suppress(json.JSONDecodeError):
                    yield json.loads(payload)
        finally:
            with contextlib.suppress(OSError):
                response.close()

    def tokenize(self, text: str | bytes) -> list[int]:
        content = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
        with self._post("/tokenize", {"content": content}) as resp:
            return [int(tok) for tok in json.loads(resp.read()).get("tokens") or []]
