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
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

from vipera import config
from vipera.backends.loading import check_host_ram, load_attempts
from vipera.backends.protocol import CHAT_PARAMS, DEFAULT_SAMPLING
from vipera.errors import BackendError
from vipera.optimizer import OptimizerError, find_llama_tool
from vipera.planner import Plan

# Loading a 14B from a cold page cache on CPU takes minutes; the HTTP server
# binds first and answers 503 until the model is ready, so poll for a while.
_START_TIMEOUT_S = 900
_HEALTH_INTERVAL_S = 0.5
_STOP_TIMEOUT_S = 10.0
#: The child's own /health is a loopback liveness check; a probe that takes
#: longer than this is already the answer.
_HEALTH_PROBE_TIMEOUT_S = 2.0
#: Floor for one request's socket timeout. It exists to break a hang, so it is
#: generous: a wedged child used to hold the gateway permit forever.
_MIN_REQUEST_TIMEOUT_S = 300.0
#: Slowest decode rate worth supporting on CPU-only hardware, in tokens/s.
#: A request is allowed ``max_tokens / this`` seconds before it counts as stuck.
_MIN_TOKENS_PER_S = 0.5
#: Non-generation calls (tokenize, embeddings) are local and bounded.
_AUX_TIMEOUT_S = 120.0


def _request_timeout(max_tokens: Any) -> float:
    """Socket timeout for one generation: a backstop, not a deadline.

    ``urlopen`` applies it per socket operation, so a stream that keeps sending
    tokens never trips it, while a child that stopped producing bytes trips it
    once instead of blocking the gateway permit forever. Long generations get a
    proportionally longer bound so slow hardware is not cut off.
    """
    try:
        tokens = max(0, int(max_tokens or 0))
    except (TypeError, ValueError):
        tokens = 0
    return max(_MIN_REQUEST_TIMEOUT_S, tokens / _MIN_TOKENS_PER_S)


def _timeout_message(path: str, timeout: float) -> str:
    return f"llama-server did not respond on {path} within {timeout:.0f}s"


# Printed by llama.cpp builds without a GPU backend. The child keeps running
# but ignores --gpu-layers, so the plan must be corrected instead of reporting
# layers that are not on the GPU.
_GPU_IGNORED_MARKER = "--gpu-layers option will be ignored"

# Chat parameters llama-server understands. It ignores unknown JSON fields, but
# sending a curated set keeps a typo from becoming a 400 in the child.
_SERVER_CHAT_PARAMS = CHAT_PARAMS - {"functions", "function_call", "grammar"}


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


def supports_n_cpu_moe(binary: str) -> bool:
    """Whether this llama-server knows `-ncmoe` / `--n-cpu-moe`.

    A build without the flag exits after allocating the model, so the planner
    must know before choosing the server backend. Some builds print the short
    form, others only the long one; accept either.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{probe.stdout}{probe.stderr}"
    return "-ncmoe" in help_text or "--n-cpu-moe" in help_text


def supports_n_cpu_ffn(binary: str) -> bool:
    """Whether this llama-server knows `-ncffn` / `--n-cpu-ffn`.

    Same contract as `supports_n_cpu_moe`: the planner must know before
    choosing the server backend, because a build without the flag exits after
    allocating the model and the child would never serve a request.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{probe.stdout}{probe.stderr}"
    return "-ncffn" in help_text or "--n-cpu-ffn" in help_text


def probe_devices(binary: str) -> str | None:
    """Return `<binary> --list-devices`, or None when the probe cannot run.

    Flag probes (`--help`) only prove what a build can parse. They do not prove
    that its GPU backend can see the card: a binary built without one, or one
    whose runtime libraries are missing from LD_LIBRARY_PATH, starts normally
    and silently ignores --gpu-layers. This is the check that catches that.
    """
    try:
        probe = subprocess.run(
            [binary, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = f"{probe.stdout}{probe.stderr}".strip()
    return output or None


# Device ids llama.cpp prints in `--list-devices`, one per backend it was built
# with. A HIP build on an AMD card still names the device CUDA0, because
# ggml-cuda serves both; SYCL names Intel GPUs, Vulkan names whatever the
# loader returned, and MTL names Metal devices.
_DEVICE_PREFIX = re.compile(r"^(?:CUDA|ROCm|HIP|SYCL|Vulkan|MTL|Metal|OpenCL|OPENCL)\d+:")


def accelerator_device(devices: str | None) -> str | None:
    """The first GPU device line in `--list-devices`, or None for CPU-only."""
    for line in (devices or "").splitlines():
        stripped = line.strip()
        if _DEVICE_PREFIX.match(stripped):
            return stripped
    return None


def has_accelerator_device(devices: str | None) -> bool:
    """Whether `--list-devices` found a GPU on any backend."""
    return accelerator_device(devices) is not None


def has_cuda_device(devices: str | None) -> bool:
    """Deprecated name for `has_accelerator_device`, kept for plugins."""
    return has_accelerator_device(devices)


def supports_spec_type(binary: str) -> bool:
    """Whether this llama-server knows `--spec-type` (the n-gram drafters).

    Older builds have neither the flag nor the n-gram drafters, and passing an
    unknown option kills the child, so the flag is only added after a probe.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--spec-type" in f"{probe.stdout}{probe.stderr}"


def _supports_fit(binary: str) -> bool:
    """Whether this llama-server knows `-fit` / `--fit`.

    Recent builds default `--fit on`, which adjusts unset arguments to leave
    `--fit-target` MiB free (1024 by default). That would override the
    planner's ngl/ncmoe placement, so Vipera turns it off — but only on builds
    that understand the flag.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{probe.stdout}{probe.stderr}"
    return "-fit" in help_text or "--fit" in help_text


def supports_embeddings(binary: str) -> bool:
    """Whether this llama-server can serve ``/v1/embeddings`` for a chat model.

    ``--embeddings`` alone is not enough: without a pooling type the endpoint
    answers 400 ("Pooling type 'none' is not OAI compatible") and a chat
    request against such a child segfaulted on the reference build, so both
    flags are required and passed together.
    """
    try:
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{probe.stdout}{probe.stderr}"
    return "--embeddings" in help_text and "--pooling" in help_text


def server_command(
    binary: str,
    model_path: str,
    plan: Plan,
    port: int,
    *,
    load_mode_supported: bool | None = None,
    fit_supported: bool | None = None,
    spec_type: str | None = None,
    spec_type_supported: bool | None = None,
    embeddings: bool = False,
    embeddings_supported: bool | None = None,
    parallel: int = 1,
) -> list[str]:
    """Translate a Plan into llama-server flags.

    `--jinja` is required for tool calling and for models whose chat template
    is not one of the hard-coded ones; it is harmless for plain chat.

    `load_mode_supported`, `fit_supported` and `spec_type_supported` skip the
    `--help` probes; `None` probes the binary.

    `spec_type` is llama-server's own speculative decoding. Measured on the
    reference 4 GB card (Hermes-4-14B Q3_K_M, 17/40 layers): `ngram-map-k`
    decodes 8.54 t/s against 6.23 t/s with it off (+37%) when the prompt is a
    file the model then rewrites, and 6.33 against 6.40 (-1%, i.e. noise) on
    fresh prose — the n-gram drafter only pays where the output repeats the
    input, and costs a hash lookup where it does not.
    """
    # A tensor-override plan (-ncmoe, -ncffn) keeps every layer on the GPU and
    # moves only the tensors named by the flag, so -ngl is the full stack rather
    # than the classic gpu_layers value (which exists for the in-process
    # fallback).
    tensor_placement = plan.moe_cpu_layers >= 0 or plan.n_cpu_ffn >= 0
    gpu_layers = plan.n_layers if tensor_placement and plan.n_layers > 0 else plan.gpu_layers
    # llama.cpp counts the output layer as one group past the GGUF's
    # block_count, so "everything on the GPU" is n_layers + 1. Measured on
    # qwen2.5-0.5B Q4_K_M (24 blocks, GTX 1650): -ngl 24 offloaded 24/25 and
    # decoded 186.7 t/s, -ngl 25 offloaded 25/25 and decoded 203.7 t/s. The
    # banner said 24/24 either way, so only llama.cpp's own "offloaded 24/25
    # layers" line showed the missing layer.
    if plan.n_layers > 0 and gpu_layers >= plan.n_layers:
        gpu_layers = plan.n_layers + 1
    # -ub is the prefill lever (measured: 896 was the best all-round point on
    # the reference 4 GB card); -b must be at least as large as -ub.
    ubatch = plan.n_ubatch or min(plan.n_batch, 512)
    batch = max(plan.n_batch, ubatch)
    command = [
        binary,
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        # -c is the whole KV window, shared by all slots, so N slots need N
        # times the window to give each request the context it planned for.
        str(plan.context * max(1, parallel)),
        "-ngl",
        str(gpu_layers),
        "-t",
        str(plan.cpu_threads),
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
        "-fa",
        "on" if plan.flash_attn else "off",
        "--jinja",
    ]
    # Always explicit: llama.cpp builds here default to 4 slots, and a slot only
    # gets 1/N of `-c`, so a plan that promised 4096 tokens would hand each
    # request 1024. Passing -np 1 is what makes `-c` mean what the planner said.
    command.extend(["-np", str(max(1, parallel))])
    if fit_supported is None:
        fit_supported = _supports_fit(binary)
    if fit_supported:
        # The planner owns placement; auto-fit would shrink it to leave 1 GiB
        # free and silently invalidate both the banner and the A/B numbers.
        command.extend(["-fit", "off"])
    if plan.moe_cpu_layers >= 0:
        command.extend(["-ncmoe", str(plan.moe_cpu_layers)])
    # -ncffn moves the dense FFN of the first N layers, leaving attention (and
    # its KV cache) on the card. Registry.resolve_backend only keeps a plan with
    # n_cpu_ffn set on the server backend when the binary has the flag, so this
    # is not probed again per spawn.
    if plan.n_cpu_ffn >= 0:
        command.extend(["-ncffn", str(plan.n_cpu_ffn)])
    if spec_type and spec_type != "none":
        supported = (
            supports_spec_type(binary) if spec_type_supported is None else spec_type_supported
        )
        if supported:
            command.extend(["--spec-type", spec_type])
    cache_k, cache_v = plan.kv_types
    if cache_k != "f16" or cache_v != "f16":
        # Both halves are named once either differs, so the command line says
        # what the banner showed instead of leaning on llama.cpp's defaults.
        command.extend(["-ctk", cache_k, "-ctv", cache_v])
    # Only when something is offloaded: with no layers on the card there is no
    # KV there to keep off.
    if gpu_layers > 0 and not plan.offload_kqv:
        command.append("--no-kv-offload")
    if plan.use_mlock:
        supported = (
            _supports_load_mode(binary) if load_mode_supported is None else load_mode_supported
        )
        if supported:
            command.extend(["-lm", "mlock"])
    if embeddings:
        supported = (
            supports_embeddings(binary) if embeddings_supported is None else embeddings_supported
        )
        if supported:
            # mean pooling is the sensible default for a chat model asked for
            # embeddings; a dedicated embedding GGUF would prefer cls.
            command.extend(["--embeddings", "--pooling", "mean"])
    return command


_GRAPHS_ENV = "GGML_CUDA_DISABLE_GRAPHS"


def server_env(plan: Plan, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the llama-server child, not for this process.

    CUDA graph capture asks for extra device memory on top of the weights and
    the KV cache, and it does so per shape: a plan picked from "did it load"
    passes health checks at a few hundred tokens of context and then takes the
    whole server down with `cudaGraphInstantiate` OOM as soon as a long prompt
    arrives. Measured on a GTX 1650 4 GB, dense 14B Q3_K_M, 8192 ctx, q8_0 KV:
    18/40 layers loaded, served short prompts at 5.72 t/s, and died with
    "CUDA error: out of memory ... in function ggml_cuda_graph_evaluate_and_capture"
    on the first ~3k-token prompt; the same plan with graphs off served that
    prompt at 37.9 t/s prefill and 3.52 t/s decode. A hybrid split means the
    plan already spent the budget (that is why layers were left on the CPU), so
    it is exactly the case at risk of that cliff; full offload keeps graphs.    Graphs are worth nothing measurable here: 17/40 with graphs on decoded
    3.51 t/s against 3.52 t/s for 18/40 with graphs off, i.e. the flag keeps a
    layer instead of costing one. An explicit value from the caller wins.

    A tensor split (--n-cpu-ffn) is the same kind of plan by that test: the
    budget went to placing every attention block, so what is left for capture is
    whatever the FFN blocks did not take. Measured on the reference 4 GB card, a
    tensor split at 3750 tokens decoded 4.97 t/s with capture on against 4.89
    t/s with it off — capture won there because that plan happened to leave
    780 MiB free, which a plan that fills the budget cannot promise. Since the
    flag is worth noise either way, hybrid plans of both kinds give it up.
    """
    env = dict(os.environ if base is None else base)
    classic_hybrid = (
        plan.moe_cpu_layers < 0 and plan.n_cpu_ffn < 0 and 0 < plan.gpu_layers < plan.n_layers
    )
    tensor_hybrid = plan.n_cpu_ffn >= 0 and 0 < plan.n_cpu_ffn < plan.n_layers
    if classic_hybrid or tensor_hybrid:
        env.setdefault(_GRAPHS_ENV, "1")
    return env


class _SseStream:
    """The child's SSE response as a blocking iterator that can be closed.

    `read1` returns whatever has already arrived instead of one SSE line, so a
    single read carries several tokens when the model is fast. `close()` is
    callable from another thread on purpose: the gateway closes it to unblock a
    reader parked inside a read when the client goes away, and only then hands
    the slot to the next request.
    """

    def __init__(self, response, *, timeout: float) -> None:
        self._response = response
        self._timeout = timeout
        self._finished = False

    def __iter__(self) -> _SseStream:
        return self

    def __next__(self) -> bytes:
        if self._finished:
            raise StopIteration
        try:
            chunk = self._response.read1(65536)
        except TimeoutError:
            self.close()
            raise RuntimeError(_timeout_message("/v1/chat/completions", self._timeout)) from None
        if not chunk:
            self.close()
            raise StopIteration
        if b"[DONE]" in chunk:
            # The last frame is already in hand; nobody will read more.
            self._finished = True
            with contextlib.suppress(OSError, AttributeError):
                self._response.close()
        return chunk

    def close(self) -> None:
        self._finished = True
        with contextlib.suppress(OSError, AttributeError):
            self._response.close()


class ServerEngine:
    """Engine-compatible client for a `llama-server` child process."""

    def __init__(self, plan: Plan, model_path: str):
        check_host_ram(plan)

        self.plan = plan
        self._model_path = model_path
        self._closed = False
        self._proc: subprocess.Popen | None = None
        self._log: BinaryIO | None = None
        self._port: int | None = None
        self._log_path: Path | None = None

        try:
            self._binary = find_llama_tool("server")
        except OptimizerError as exc:
            raise RuntimeError(
                f"{exc} The 'server' backend needs the llama-server binary; "
                "the Python binding does not ship one."
            ) from exc

        # Probe once per engine: the load ladder may spawn the child several
        # times, and each extra --help run is wasted startup time.
        self._fit_supported = _supports_fit(self._binary)
        cfg = config.load_config()
        # Slots: >1 lets the gateway run that many requests at once through
        # llama.cpp's continuous batching (see build_app's semaphore).
        self.parallel = max(1, int(cfg.get("parallel", 1) or 1))
        self._spec_type = cfg.get("spec_type", "none")
        self._spec_type_supported = supports_spec_type(self._binary)
        self.supports_embeddings = bool(cfg.get("embeddings", False)) and supports_embeddings(
            self._binary
        )
        if cfg.get("embeddings", False) and not self.supports_embeddings:
            print(
                "[!] Warning: this llama-server has no --embeddings/--pooling; "
                "/v1/embeddings will be unavailable."
            )

        # Same staged fallback as the in-process engine: walk the offload down
        # before giving up the placement entirely, because CPU-only is the
        # slowest outcome and a one-layer-too-optimistic plan pays a few
        # seconds for the retry instead of the whole run. With `moe_probe` on,
        # a MoE plan tries a more aggressive expert placement first and keeps it
        # only if it starts: the fit the planner computed is a floor on what the
        # card can take, not a measurement (see loading.load_attempts).
        probe = bool(cfg.get("moe_probe", True))
        last_exc: Exception | None = None
        fell_back = False
        for candidate, label in load_attempts(plan, probe=probe):
            try:
                self._start(candidate)
            except RuntimeError as exc:
                last_exc = exc
                fell_back = True
                continue
            if candidate != plan:
                if fell_back:
                    print(
                        "\n[!] Warning: llama-server failed to start with the planned "
                        f"split; serving {label}."
                    )
                else:
                    print(f"\n[i] llama-server started with {label}.")
            break
        else:
            raise RuntimeError(
                f"llama-server failed to start on GPU and CPU: {last_exc}"
            ) from last_exc

        atexit.register(self.close)

    # -- lifecycle ---------------------------------------------------------

    def _start(self, plan: Plan) -> None:
        port = _free_port()
        log_path = config.LOGS_DIR / f"llama-server-{port}.log"
        command = server_command(
            self._binary,
            self._model_path,
            plan,
            port,
            fit_supported=self._fit_supported,
            spec_type=self._spec_type,
            spec_type_supported=self._spec_type_supported,
            embeddings=self.supports_embeddings,
            embeddings_supported=self.supports_embeddings,
            parallel=self.parallel,
        )
        # The handle is handed to Popen and must outlive this function.
        handle = open(log_path, "wb")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=server_env(plan),
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
                moe_cpu_layers=-1,
                n_ubatch=0,
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

    def _post(self, path: str, body: dict, *, timeout: float):
        request = urllib.request.Request(
            self._base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"llama-server {path} returned {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llama-server is unreachable: {exc.reason}") from None
        except TimeoutError:
            raise RuntimeError(_timeout_message(path, timeout)) from None

    @staticmethod
    def _body(resp, path: str, timeout: float) -> bytes:
        """Read a response body, reporting a child that stopped sending."""
        try:
            return resp.read()
        except TimeoutError:
            raise RuntimeError(_timeout_message(path, timeout)) from None

    def health(self) -> str:
        """Ask the child's own ``/health``; never starts or restarts it.

        ``ok`` only when the child answered 200. Otherwise the state a
        readiness probe needs: ``starting`` (503 while it loads), ``exited``
        (the process is gone) or ``unreachable`` (no answer on its port).
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return "exited"
        try:
            url = f"{self._base_url}/health"
        except RuntimeError:
            return "exited"
        try:
            with urllib.request.urlopen(url, timeout=_HEALTH_PROBE_TIMEOUT_S) as resp:
                return "ok" if resp.status == 200 else "starting"
        except urllib.error.HTTPError as exc:
            return "starting" if exc.code == 503 else "unreachable"
        except (urllib.error.URLError, OSError):
            # Includes the socket timeout: no answer is no answer.
            return "unreachable"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in the order they were given."""
        if not self.supports_embeddings:
            raise BackendError(
                "this llama-server was not started with embedding support; "
                "set `embeddings true` in the config and restart"
            )
        with self._post("/v1/embeddings", {"input": texts}, timeout=_AUX_TIMEOUT_S) as resp:
            payload = json.loads(self._body(resp, "/v1/embeddings", _AUX_TIMEOUT_S))
        items = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
        return [list(item["embedding"]) for item in items]

    def chat(self, messages: list[dict], stream: bool = False, **kw: Any):
        params = {k: v for k, v in kw.items() if k in _SERVER_CHAT_PARAMS and v is not None}
        for key, default in DEFAULT_SAMPLING.items():
            params.setdefault(key, default)
        body: dict[str, Any] = {"messages": messages, "stream": stream, **params}
        timeout = _request_timeout(params.get("max_tokens"))
        # Not part of the in-process engine's surface, but llama-server uses it
        # to put the exact token usage in the final SSE chunk; bench relies on
        # that for an exact generation count.
        if kw.get("stream_options") is not None:
            body["stream_options"] = kw["stream_options"]

        if not stream:
            with self._post("/v1/chat/completions", body, timeout=timeout) as resp:
                return json.loads(self._body(resp, "/v1/chat/completions", timeout))

        # A streaming generator keeps the HTTP response open across yields, so
        # the caller can consume it exactly like llama.cpp's own generator.
        response = self._post("/v1/chat/completions", body, timeout=timeout)
        return self._iter_sse(response, timeout=timeout)

    #: The gateway may forward this engine's SSE bytes verbatim instead of
    #: parsing and re-serialising every chunk in Python (see `chat_sse`).
    passthrough_sse = True

    def chat_sse(self, messages: list[dict], **kw: Any) -> _SseStream:
        """Return llama-server's SSE frames unchanged, as the client sent them.

        The child already speaks the OpenAI wire format, so a gateway that
        re-encodes each frame in Python pays for parsing and `json.dumps` on
        every token for nothing. This path keeps the bytes.
        """
        params = {k: v for k, v in kw.items() if k in _SERVER_CHAT_PARAMS and v is not None}
        for key, default in DEFAULT_SAMPLING.items():
            params.setdefault(key, default)
        body: dict[str, Any] = {"messages": messages, "stream": True, **params}
        if kw.get("stream_options") is not None:
            body["stream_options"] = kw["stream_options"]
        timeout = _request_timeout(params.get("max_tokens"))
        return _SseStream(
            self._post("/v1/chat/completions", body, timeout=timeout), timeout=timeout
        )

    @staticmethod
    def _iter_sse(response, *, timeout: float = _MIN_REQUEST_TIMEOUT_S) -> Iterator[dict]:
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
        except TimeoutError:
            raise RuntimeError(_timeout_message("/v1/chat/completions", timeout)) from None
        finally:
            with contextlib.suppress(OSError):
                response.close()

    def tokenize(self, text: str | bytes) -> list[int]:
        content = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
        with self._post("/tokenize", {"content": content}, timeout=_AUX_TIMEOUT_S) as resp:
            body = self._body(resp, "/tokenize", _AUX_TIMEOUT_S)
        return [int(tok) for tok in json.loads(body).get("tokens") or []]
