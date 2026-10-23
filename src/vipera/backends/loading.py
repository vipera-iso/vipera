"""Plan fallback logic: how much to offload, and in what order to retry.

Kept free of any engine import so both backends (and the planner's callers) can
use it without pulling in llama.cpp or a subprocess.
"""

from __future__ import annotations

from dataclasses import replace

import psutil

from vipera.errors import BackendError
from vipera.planner import Plan


def check_host_ram(plan: Plan) -> None:
    """Refuse a plan whose host-side footprint clearly exceeds free RAM.

    An OOM here is an uncatchable SIGKILL, not a Python exception, so the
    GPU->CPU fallback never gets a chance to run. Shared by both backends: the
    llama-server child dies the same way.
    """
    if plan.ram_estimate_gb <= 0:
        return
    available_gb = psutil.virtual_memory().available / 1024**3
    if plan.ram_estimate_gb > available_gb + 0.5:
        raise BackendError(
            f"Not enough RAM for this plan: ~{plan.ram_estimate_gb:.1f} GB "
            f"needed on the host, {available_gb:.1f} GB available. "
            f"Lower --ctx, use kv_cache=q4_0, or close other applications."
        )


def layer_ladder(layers: int) -> list[int]:
    """Offload counts to try for a plan that asks for ``layers``, best first.

    A split that is one or two layers too optimistic used to fall straight to
    ``gpu_layers=0``, which is the slowest outcome on the box this was measured
    on: 3.90 t/s CPU-only against 4.97 t/s at the last split that loaded
    (14B/Q4, 8K, GTX 1650). Each rung costs one failed context creation (~2 s
    for a 14B at 8K), and a failed attempt does not hand its VRAM back before
    the next one: after three failures on a 4 GB card even a 10-layer rung
    failed, which is how asking for 20 layers ended up on the CPU. The ladder is
    therefore short and its last rung is a quarter of the plan, not half — it
    exists to land anywhere other than CPU-only.
    """
    if layers <= 0:
        return []
    steps = (layers, layers - 1, layers - 2, max(2, layers // 4))
    return [n for n in dict.fromkeys(steps) if 0 < n <= layers]


def with_gpu_layers(plan: Plan, layers: int) -> Plan:
    """Same plan at a different offload depth, with a truthful note."""
    note = plan.note
    if note.startswith("Hybrid: "):
        total = note.split("/", 1)[1].split()[0]
        note = f"Hybrid: {layers}/{total} layers"
    elif note == "Full GPU":
        # A large card can plan a full offload and still fail to create the
        # context (the driver holds more than the weights). Reporting "Full
        # GPU" while running a reduced split would repeat the lie the ladder
        # exists to stop.
        note = f"Hybrid: {layers}/all layers"
    return replace(plan, gpu_layers=layers, note=note)


def load_attempts(plan: Plan) -> list[tuple[Plan, str]]:
    """Candidate plans for loading, most offload first, ending CPU-only."""
    attempts: list[tuple[Plan, str]] = [(plan, "planned split")]
    if plan.requires_server and plan.n_layers > 0:
        # The server ladder moves one expert layer at a time to the CPU, which
        # frees roughly 190-220 MB on the reference model. The last rung keeps
        # every expert on the CPU instead of failing the load entirely.
        rungs = (plan.moe_cpu_layers + 1, plan.moe_cpu_layers + 2, plan.n_layers)
        for cpu_layers in dict.fromkeys(rungs):
            if cpu_layers <= plan.moe_cpu_layers or cpu_layers > plan.n_layers:
                continue
            gpu_experts = plan.n_layers - cpu_layers
            attempts.append(
                (
                    replace(
                        plan,
                        moe_cpu_layers=cpu_layers,
                        note=f"MoE: {gpu_experts}/{plan.n_layers} expert layers on GPU",
                    ),
                    f"MoE, {gpu_experts}/{plan.n_layers} expert layers",
                )
            )
    else:
        hybrid_total = (
            plan.note.split("/", 1)[1].split()[0] if plan.note.startswith("Hybrid: ") else None
        )
        for layers in layer_ladder(plan.gpu_layers)[1:]:
            label = f"GPU, {layers}/{hybrid_total} layers" if hybrid_total else f"GPU, {layers}"
            attempts.append((with_gpu_layers(plan, layers), label))

    # CPU stages: keep the caller's context and KV type first, then reduce.
    # Clearing the MoE fields matters: -ncmoe with -ngl 0 is not a valid
    # CPU-only plan.
    cpu_plan = replace(
        plan,
        gpu_layers=0,
        offload_kqv=False,
        n_batch=min(plan.n_batch, 128),
        moe_cpu_layers=-1,
        n_ubatch=0,
    )
    reduced_plan = replace(cpu_plan, context=min(plan.context, 2048))
    safe_plan = replace(reduced_plan, kv_cache="f16", flash_attn=False)
    attempts.append((cpu_plan, f"CPU, ctx={cpu_plan.context}, kv={cpu_plan.kv_cache}"))
    attempts.append((reduced_plan, f"CPU, ctx={reduced_plan.context}, kv={reduced_plan.kv_cache}"))
    attempts.append((safe_plan, f"CPU safe mode, ctx={safe_plan.context}, kv={safe_plan.kv_cache}"))
    return attempts
