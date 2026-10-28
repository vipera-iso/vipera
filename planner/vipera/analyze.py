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


# Block-quantized KV types do not cost one byte per element: q8_0 packs 32 values
# into 34 bytes. Counting the nominal width would under-report the cache by 6%,
# which is the same order as the headroom this budget is meant to protect.
KV_BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}


def kv_cache_bytes(gguf: GGUF, context: int, *, kv_type: str = "f16") -> int | None:
    """Bytes the KV cache needs for `context` tokens, read off the GGUF header.

    Returns None when the header does not describe the cache — MLA checkpoints
    keep a latent cache instead of per-head K/V, and a guessed number would be
    worse than none, because the caller budgets RAM with it. Validated against
    llama.cpp on OLMoE-1B-7B Q4_K_M: 512 MiB at 4096 tokens, which is the size
    the runtime logs for the same file.
    """
    arch = gguf.architecture
    md = gguf.metadata
    if not arch or context <= 0:
        return None
    per_element = KV_BYTES_PER_ELEMENT.get(kv_type)
    n_layer = md.get(f"{arch}.block_count")
    heads_kv = md.get(f"{arch}.attention.head_count_kv")
    key_len = md.get(f"{arch}.attention.key_length")
    value_len = md.get(f"{arch}.attention.value_length") or key_len
    if key_len is None:
        n_embd = md.get(f"{arch}.embedding_length")
        head_count = md.get(f"{arch}.attention.head_count")
        if isinstance(n_embd, int) and isinstance(head_count, int) and head_count > 0:
            key_len = value_len = n_embd // head_count
    if per_element is None or not isinstance(n_layer, int):
        return None
    if isinstance(heads_kv, int):
        heads = [heads_kv] * n_layer
    elif isinstance(heads_kv, list):
        heads = [h for h in heads_kv if isinstance(h, int)]
    else:
        return None
    if not isinstance(key_len, int) or not isinstance(value_len, int):
        return None
    per_token = sum(h * (key_len + value_len) for h in heads)
    return int(per_token * per_element * context)


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
