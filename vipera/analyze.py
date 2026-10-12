"""Turn a GGUF descriptor list into an inventory vipera can plan against.

The only job here is to answer: how many bytes are *always* needed (attention,
norms, embeddings, router, shared experts) versus how many bytes belong to
routable experts that can be cached, spilled, or reordered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .gguf import GGUF, TensorInfo

_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def _layer_of(name: str) -> int | None:
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def is_expert_tensor(name: str) -> bool:
    """Routable expert weights (not shared experts, not the router)."""
    return "_exps" in name and "_shexp" not in name


def is_shared_expert(name: str) -> bool:
    return "_shexp" in name


def is_router(name: str) -> bool:
    # llama.cpp names the MoE gate/router `ffn_gate_inp`.
    return name.endswith("ffn_gate_inp.weight")


def n_experts_of(t: TensorInfo) -> int | None:
    """Expert count from a 3-D expert tensor shaped (n_embd, n_ff, n_experts)."""
    if len(t.shape) >= 3:
        return int(t.shape[2])
    return None


@dataclass
class LayerInventory:
    index: int
    resident_bytes: int = 0
    expert_bytes: int = 0
    router_bytes: int = 0
    shared_expert_bytes: int = 0
    n_experts: int | None = None
    expert_tensors: list[str] = field(default_factory=list)


@dataclass
class Inventory:
    architecture: str | None
    n_layers: int | None
    resident_bytes: int
    expert_bytes: int
    router_bytes: int
    shared_expert_bytes: int
    other_bytes: int
    n_experts: int | None
    layers: list[LayerInventory]
    tensors: list[TensorInfo]

    @property
    def total_bytes(self) -> int:
        return self.resident_bytes + self.expert_bytes + self.router_bytes + self.shared_expert_bytes + self.other_bytes

    @property
    def always_resident_bytes(self) -> int:
        """Everything that must be addressable on every token: not routable."""
        return self.resident_bytes + self.router_bytes + self.shared_expert_bytes + self.other_bytes


def _block_count(gguf: GGUF) -> int | None:
    arch = gguf.architecture
    if arch:
        v = gguf.metadata.get(f"{arch}.block_count")
        if isinstance(v, int):
            return v
    # Fall back to the highest blk index seen.
    idx = [i for i in (_layer_of(t.name) for t in gguf.tensors) if i is not None]
    return (max(idx) + 1) if idx else None


def analyze(gguf: GGUF) -> Inventory:
    layers: dict[int, LayerInventory] = {}
    resident = expert = router = shared = other = 0
    all_expert_counts: list[int] = []

    for t in gguf.tensors:
        size = t.n_bytes
        idx = _layer_of(t.name)
        layer = layers.setdefault(idx, LayerInventory(index=idx)) if idx is not None else None

        if is_expert_tensor(t.name):
            expert += size
            if layer is not None:
                layer.expert_bytes += size
                layer.expert_tensors.append(t.name)
                n = n_experts_of(t)
                if n is not None:
                    layer.n_experts = n
                    all_expert_counts.append(n)
        elif is_router(t.name):
            router += size
            if layer is not None:
                layer.router_bytes += size
        elif is_shared_expert(t.name):
            shared += size
            if layer is not None:
                layer.shared_expert_bytes += size
        elif idx is not None:
            resident += size
            if layer is not None:
                layer.resident_bytes += size
        else:
            # token_embd, output, output_norm, ...
            other += size

    n_experts = None
    if all_expert_counts:
        # Every routed tensor on every layer should agree; report the mode.
        n_experts = max(set(all_expert_counts), key=all_expert_counts.count)

    return Inventory(
        architecture=gguf.architecture,
        n_layers=_block_count(gguf),
        resident_bytes=resident,
        expert_bytes=expert,
        router_bytes=router,
        shared_expert_bytes=shared,
        other_bytes=other,
        n_experts=n_experts,
        layers=sorted(layers.values(), key=lambda l: (-1 if l.index is None else l.index)),
        tensors=gguf.tensors,
    )


def to_dict(inv: Inventory) -> dict:
    return {
        "architecture": inv.architecture,
        "n_layers": inv.n_layers,
        "n_experts": inv.n_experts,
        "bytes": {
            "resident": inv.resident_bytes,
            "router": inv.router_bytes,
            "shared_expert": inv.shared_expert_bytes,
            "other": inv.other_bytes,
            "expert": inv.expert_bytes,
            "always_resident": inv.always_resident_bytes,
            "total": inv.total_bytes,
        },
        "layers": [
            {
                "index": l.index,
                "resident": l.resident_bytes,
                "expert": l.expert_bytes,
                "router": l.router_bytes,
                "shared_expert": l.shared_expert_bytes,
                "n_experts": l.n_experts,
                "expert_tensors": len(l.expert_tensors),
            }
            for l in inv.layers
        ],
    }
