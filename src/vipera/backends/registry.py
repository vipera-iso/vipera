"""Choosing a backend that can actually run a plan, and building it.

The server module is imported once and reached through attributes (rather than
importing its functions by name) so a probe can be substituted in tests and,
later, replaced by a plugin lookup without changing this file.
"""

from __future__ import annotations

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
    the only safe fallback). If a llama-server with ``-ncmoe`` is available,
    switch to it; otherwise clear the MoE fields so the banner and bench report
    what is really running.
    """
    # The MoE upgrade below only makes sense for llama.cpp's two backends.
    if backend != "inprocess" or not plan.requires_server:
        return backend, plan

    try:
        binary = find_llama_tool("server")
        supported = server_backend.supports_n_cpu_moe(binary)
    except (OptimizerError, OSError):
        binary = None
        supported = False

    if supported:
        gpu_experts = plan.n_layers - plan.moe_cpu_layers
        print(
            "[i] MoE plan: using llama-server for expert placement "
            f"({gpu_experts}/{plan.n_layers} expert layers on GPU, "
            f"ubatch {plan.n_ubatch})."
        )
        return "server", plan

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
