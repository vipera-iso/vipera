"""Choosing a backend that can actually run a plan, and building it.

The server module is imported once and reached through attributes (rather than
importing its functions by name) so a probe can be substituted in tests and,
later, replaced by a plugin lookup without changing this file.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from vipera.backends import server as server_backend
from vipera.backends.inprocess import Engine
from vipera.backends.names import GROUP, names
from vipera.backends.protocol import ChatEngine
from vipera.errors import BackendError
from vipera.optimizer import OptimizerError, find_llama_tool
from vipera.planner import Plan


def resolve_backend(plan: Plan, backend: str) -> tuple[str, Plan]:
    """Return a backend that can actually execute ``plan``.

    llama-cpp-python has no ``n_cpu_moe`` argument, so a MoE plan is silently
    ignored by the in-process engine (and its classic ``gpu_layers`` split is
    the only safe fallback). A llama-server with ``-ncmoe`` can run the plan,
    but only if its CUDA backend can actually see a device. A CPU-only build
    (or a CUDA build with an incomplete LD_LIBRARY_PATH) parses every flag and
    then ignores ``--gpu-layers``, which made the "upgrade" slower than the
    in-process split on the reference box. Probe devices before switching.
    """
    # The MoE upgrade below only makes sense for llama.cpp's two backends.
    if backend != "inprocess" or not plan.requires_server:
        return backend, plan

    try:
        binary = find_llama_tool("server")
        supported = server_backend.supports_n_cpu_moe(binary)
        devices = server_backend.probe_devices(binary) if supported else None
    except (OptimizerError, OSError):
        binary = None
        supported = False
        devices = None

    if supported and server_backend.has_cuda_device(devices):
        gpu_experts = plan.n_layers - plan.moe_cpu_layers
        print(
            "[i] MoE plan: using llama-server for expert placement "
            f"({gpu_experts}/{plan.n_layers} expert layers on GPU, "
            f"ubatch {plan.n_ubatch})."
        )
        return "server", plan

    if supported:
        probe = devices if devices is not None else "(probe failed)"
        probe_lines = "\n".join(f"      {line}" for line in probe.splitlines())
        ld_path = os.environ.get("LD_LIBRARY_PATH") or "(unset)"
        bin_env = os.environ.get("VIPERA_LLAMA_BIN") or "(unset)"
        print(
            "[!] Warning: llama-server parsed -ncmoe but does not see a CUDA0 "
            "device; refusing the MoE server plan and falling back to the "
            "classic layer split.\n"
            f"    binary: {binary}\n"
            f"    --list-devices:\n{probe_lines}\n"
            f"    VIPERA_LLAMA_BIN={bin_env}\n"
            f"    LD_LIBRARY_PATH={ld_path}\n"
            "    If this is a CUDA build, point LD_LIBRARY_PATH at its CUDA "
            "runtime libraries (libcudart/libcublas) and re-run."
        )
        return "inprocess", replace(plan, moe_cpu_layers=-1, n_ubatch=0)

    where = f"{binary} has no -ncmoe" if binary else "llama-server was not found"
    print(
        f"[!] Warning: {where}; the MoE plan cannot run in-process and is "
        "falling back to the classic layer split. Install a newer llama-server "
        "to use expert offload."
    )
    return "inprocess", replace(plan, moe_cpu_layers=-1, n_ubatch=0)


def _load_plugin(name: str):
    """Return the factory a plugin registered for ``name``.

    The contract is one callable: ``factory(plan, model_path) -> ChatEngine``.
    Planning stays with the built-in planner until a second real backend
    proves what a backend-specific plan needs.
    """
    from importlib.metadata import entry_points

    found = entry_points()
    # requires-python >= 3.11, so entry_points() always has .select.
    selected = found.select(group=GROUP)
    for entry in selected:
        if entry.name == name:
            return entry.load()
    raise BackendError(f"unknown backend '{name}' (known: {', '.join(names())})")


def create_engine(plan: Plan, model_path: str | Path, backend: str = "inprocess") -> ChatEngine:
    """Build the engine selected by the ``backend`` config key."""
    backend, plan = resolve_backend(plan, backend)
    path = str(model_path)
    if backend == "server":
        return server_backend.ServerEngine(plan, path)
    if backend == "inprocess":
        return Engine(plan, path)
    return _load_plugin(backend)(plan, path)
