"""Inspect GGUF headers, tensor metadata, and per-layer size distribution.

The analyzer only reads the GGUF index. Tensor payloads stay on disk, so this
is safe to run on multi-gigabyte models and does not need llama.cpp.
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_QUANT_RE = re.compile(
    r"(IQ\d+_\w+|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|NVFP4|MXFP4|TQ\d_0|BF16|F16|F32|Q\d+)",
    re.IGNORECASE,
)

_QUANT_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    40: "NVFP4",
    41: "Q1_0",
    # Not present in gguf 0.19 (the reader rejects such files); kept so a raw
    # header dump can still label the id llama.cpp reserves for it.
    42: "Q2_0",
}

# Rough bits-per-weight ordering. Only used to flag quality-sensitive tensors
# quantized below Q4_K; it is a heuristic, not a quality guarantee.
_QUANT_RANK = {
    "F64": 64.0,
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_K": 8.0,
    "Q8_0": 8.0,
    "Q8_1": 8.0,
    "Q6_K": 6.0,
    "Q5_K": 5.0,
    "Q5_0": 5.0,
    "Q5_1": 5.0,
    "Q4_K": 4.5,
    "Q4_0": 4.0,
    "Q4_1": 4.0,
    "IQ4_NL": 4.0,
    "IQ4_XS": 4.0,
    "Q3_K": 3.0,
    "IQ3_XXS": 3.0,
    "IQ3_S": 3.0,
    "Q2_K": 2.0,
    "IQ2_XXS": 2.0,
    "IQ2_XS": 2.0,
    "IQ2_S": 2.0,
    "IQ1_S": 1.0,
    "IQ1_M": 1.0,
    "Q2_0": 2.0,
    "TQ2_0": 2.2,
    "TQ1_0": 1.7,
    "Q1_0": 1.0,
    # 4-bit float formats; ranked with the other 4-bit types, below Q4_K.
    "MXFP4": 4.0,
    "NVFP4": 4.0,
}

# Tensor names that imply a layer keeps a KV cache, and names that imply a
# recurrent/SSM layer with a fixed-size state instead. Hybrid stacks (KDA/BCSA,
# Mamba, RWKV) mix both; the planner only needs the split.
# `attn_k` also covers MLA/KDA-adjacent names: attn_k_b, attn_kv_a_mqa, attn_k_norm.
_KV_TENSOR_MARKERS = ("attn_k", "attn_v", "attn_qkv")
# `conv` also covers `conv1d` and the `shortconv.*` stacks LFM2.5 uses for its
# non-KV layers: those layers hold no K/V tensors at all, so without the marker
# they are skipped, `hybrid` never trips and the planner prices all layers as
# KV-bearing.
_RECURRENT_TENSOR_MARKERS = ("ssm_", "linear_attn", "kda", "mamba", "conv")

_CRITICAL_TENSORS = ("ffn_down", "attn_v")
_CRITICAL_THRESHOLD = "Q4_K"
_CRITICAL_RANK = _QUANT_RANK[_CRITICAL_THRESHOLD]

# Quant type ids the static table may lag behind: gguf grows a new type every
# few releases, and labelling an unknown id as TYPE_nn turns into a bogus
# quality warning because it ranks 0. Look the name up in gguf when possible.
_GGUF_QUANT_TYPES: dict[int, str] | None = None


class GGUFAnalysisError(RuntimeError):
    """Raised when a GGUF file cannot be inspected."""


#: A GGUF header can claim any layer count, and a tensor name any layer index;
#: both size per-layer lists when the file is read. The bound is well above any
#: real stack (the largest shipping models are under 200 layers) because the
#: value is untrusted input, not a description of this machine.
_MAX_LAYERS = 1024

#: Critical-tensor warnings are per tensor, and a crafted file can hold as many
#: as it likes; the UI only ever shows a handful.
_MAX_WARNINGS = 100


def _gguf_quant_names() -> dict[int, str]:
    """Map id -> name from the optional `gguf` package, cached on first use."""
    global _GGUF_QUANT_TYPES
    if _GGUF_QUANT_TYPES is None:
        try:
            from gguf.constants import GGMLQuantizationType
        except Exception:  # package missing, or an older layout
            # Do not cache the failure: a later call (real gguf installed, or a
            # test that stubs the module) should still be able to resolve ids.
            return {}
        _GGUF_QUANT_TYPES = {int(t): t.name for t in GGMLQuantizationType}
    return _GGUF_QUANT_TYPES


def _quant_name(tensor_type: Any) -> str:
    try:
        value = int(tensor_type)
    except (TypeError, ValueError):
        return str(tensor_type)
    name = _QUANT_NAMES.get(value)
    if name:
        return name
    return _gguf_quant_names().get(value, f"TYPE_{value}")


def _quant_rank(name: str) -> float | None:
    """Rough bits-per-weight for a quant name; None when it is unknown.

    Unknown must not rank as 0: that would flag every future 4-bit type as
    "below Q4_K" and make the warning meaningless.
    """
    if name in _QUANT_RANK:
        return _QUANT_RANK[name]
    if name.startswith("IQ4"):
        return 4.0
    if name.startswith("IQ3"):
        return 3.0
    if name.startswith("IQ2"):
        return 2.0
    if name.startswith("IQ1"):
        return 1.0
    return None


def _classify_tensor(name: str) -> str:
    if name == "output.weight" or name.startswith("token_embd") or name == "token_types.weight":
        return "embedding"
    if "_norm" in name:
        return "norm"
    if "_exps" in name or "ffn_gate_inp" in name or "expert" in name:
        return "expert"
    if "attn_" in name:
        return "attention"
    if "ffn_" in name:
        return "ffn"
    return "other"


def _layer_index(name: str) -> int | None:
    match = re.match(r"blk\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _attention_profile(tensor_names: list[str], n_layers: int | None) -> dict[str, Any]:
    """Split layers into KV-cache layers and recurrent (state-only) layers.

    Detection is name-based because GGUF metadata does not describe per-layer
    attention type: KDA/Mamba layers carry `ssm_*` / `linear_attn` / `conv1d`
    tensors, attention layers carry `attn_k`/`attn_v`/`attn_qkv`. A layer that
    matches both counts as KV-bearing, since over-reserving KV is the safe side
    of a memory budget.
    """
    kv_layers: set[int] = set()
    recurrent_layers: set[int] = set()
    for name in tensor_names:
        layer = _layer_index(name)
        if layer is None:
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in _RECURRENT_TENSOR_MARKERS):
            recurrent_layers.add(layer)
        elif any(marker in lowered for marker in _KV_TENSOR_MARKERS):
            kv_layers.add(layer)

    recurrent_layers -= kv_layers
    total = int(n_layers or 0) or (max(kv_layers | recurrent_layers, default=-1) + 1)
    hybrid = bool(recurrent_layers) and bool(kv_layers)
    return {
        "hybrid_attention": hybrid,
        # Only shrink the budget when the split is actually understood; a
        # partial match on a dense model must not reduce the KV estimate.
        "kv_layers": len(kv_layers) if hybrid else total,
        "recurrent_layers": len(recurrent_layers),
    }


def _field_value(fields: dict, key: str) -> Any:
    field = fields.get(key)
    if field is None:
        return None
    try:
        value = field.contents()
    except Exception:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        item = value[0]
        if hasattr(item, "item"):
            item = item.item()
        return item
    return value


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field_first_int(fields: dict, key: str) -> int | None:
    """Return the representative element of an int-array field, or None.

    Per-layer arrays such as attention.head_count are uniform for the
    architectures we support, so the first entry describes the model. Hybrid
    stacks are the exception: lfm2moe writes ``attention.head_count_kv`` as
    0 on the conv-only layers and 8 on the attention ones, and taking the first
    entry reports zero KV heads, which makes the planner fall back to the full
    head count and price every layer as 4x its real KV. First non-zero entry
    wins; an all-zero array stays zero so the caller can fall back as before.
    """
    value = _field_value(fields, key)
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _as_int(item)
            if parsed:
                return parsed
        return _as_int(value[0]) if value else None
    return _as_int(value)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# A split GGUF names its shards `...-00001-of-00002.gguf`. llama.cpp opens
# shard 1 and follows `split.count` itself, so a file that *works* is not
# evidence that the numbers describing it are complete.
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _shard_paths(model_path: Path) -> list[Path]:
    """Every shard of a split GGUF, in order; ``[model_path]`` otherwise.

    Discovery is by filename, because that is what llama.cpp relies on too.
    Symlinks are resolved first: `vipera model add` links shard 1 into the model
    store under a name of its own, and the pattern only survives on the target.

    A declared shard count larger than the siblings on disk keeps whatever is
    there: a partially downloaded split must still analyze, and the caller can
    see the shortfall in ``split_count`` against ``split_declared``.
    """
    real_path = model_path.resolve()
    match = _SHARD_RE.search(real_path.name)
    if match is None or int(match.group(2)) <= 1:
        return [model_path]
    prefix = real_path.name[: match.start()]
    siblings = sorted(real_path.parent.glob(f"{prefix}-*-of-{match.group(2)}.gguf"))
    return siblings or [model_path]


def analyze_gguf(path: str) -> dict[str, Any]:
    """Return metadata, tensor map, and size breakdown for a GGUF model.

    A split model is analyzed **whole**: every shard is read and the per-layer
    tables are summed, so ``expert_bytes`` and ``file_size_bytes`` describe the
    same thing. They did not before, and on a two-shard 58 GB model the planner
    compared a whole-file size against one shard's expert bytes: it saw 14.5 GB
    of "non-expert" weights where there are 2.6 GB, concluded the dense side
    could never fit in VRAM, and planned the model CPU-only.

    The return value is JSON-friendly except for integer layer keys, which are
    kept as integers for convenient sorting. It raises ``GGUFAnalysisError``
    with an actionable message when the file or the optional ``gguf`` package
    is unavailable.
    """
    model_path = Path(path).expanduser()
    shards = _shard_paths(model_path)
    if len(shards) == 1:
        result = _analyze_single(str(model_path))
        declared = _declared_shards(model_path)
        if declared > 1:
            # A split whose later shards are not on disk yet still analyzes, but
            # says so: the sums below describe what is here, not the model.
            result["metadata"]["split_count"] = 1
            result["metadata"]["split_declared"] = declared
        return result
    return _merge_analysis([_analyze_single(str(shard)) for shard in shards], model_path)


def _declared_shards(model_path: Path) -> int:
    """The shard count the file name declares, or 1 for a single-file model."""
    match = _SHARD_RE.search(model_path.resolve().name)
    return int(match.group(2)) if match else 1


def _merge_analysis(parts: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    """One analysis for a split model, summed across its shards.

    Descriptions come from the first shard (every shard carries the same
    header), sizes are summed, and the per-layer lists are added index by
    index, which is what makes ``expert_layer_bytes`` cover the layers that
    live in the later shards.
    """
    first = parts[0]
    meta = dict(first["metadata"])

    def _add(key: str) -> int:
        return sum(int(part["metadata"].get(key) or 0) for part in parts)

    for key in (
        "file_size_bytes",
        "expert_bytes",
        "shared_expert_bytes",
        "attn_bytes",
        "fixed_bytes",
        "tensor_count",
    ):
        meta[key] = _add(key)
    # Not additive: `general.parameter_count` describes the whole model, and
    # gguf-split copies the entire KV block into every shard (`gguf_set_kv` in
    # tools/gguf-split). Summing it reported a 2-shard 70B as 140B.
    meta["n_params"] = max(
        (int(part["metadata"].get("n_params") or 0) for part in parts), default=0
    )

    for list_key in ("expert_layer_bytes", "ffn_layer_bytes"):
        width = max(len(part["metadata"].get(list_key) or []) for part in parts)
        meta[list_key] = [
            sum(int((part["metadata"].get(list_key) or [0] * width)[i] or 0) for part in parts)
            for i in range(width)
        ]
    meta["expert_layers"] = sum(1 for value in meta["expert_layer_bytes"] if value > 0)
    meta["split_count"] = len(parts)

    quant_sizes: dict[str, int] = defaultdict(int)
    quant_counts: dict[str, int] = defaultdict(int)
    category_sizes: dict[str, int] = defaultdict(int)
    layer_sizes: dict[int | None, int] = defaultdict(int)
    warnings: list[dict[str, str]] = []
    for part in parts:
        for key, value in part["quant_sizes"].items():
            quant_sizes[key] += int(value)
        for key, value in part["quant_counts"].items():
            quant_counts[key] += int(value)
        for key, value in part["category_sizes"].items():
            category_sizes[key] += int(value)
        for key, value in part["layer_distribution"].items():
            layer_sizes[key] += int(value)
        warnings.extend(part["warnings"])
    if quant_sizes:
        meta["dominant_quant"] = max(quant_sizes, key=lambda key: quant_sizes[key])
    truncated = (
        any(part.get("warnings_truncated") for part in parts) or len(warnings) > _MAX_WARNINGS
    )

    return {
        "path": str(model_path),
        "metadata": meta,
        "tensor_map": [tensor for part in parts for tensor in part["tensor_map"]],
        "layer_distribution": dict(
            sorted(layer_sizes.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
        ),
        "category_sizes": dict(category_sizes),
        "quant_sizes": dict(quant_sizes),
        "quant_counts": dict(quant_counts),
        "warnings": warnings[:_MAX_WARNINGS],
        "warnings_truncated": truncated,
    }


def _analyze_single(path: str) -> dict[str, Any]:
    """Metadata, tensor map, and size breakdown for one GGUF file."""
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        raise GGUFAnalysisError(f"file not found: {model_path}")

    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise GGUFAnalysisError(
            'gguf is not installed. Install it with: python -m pip install "vipera-ai[gguf]"'
        ) from exc

    try:
        reader = GGUFReader(str(model_path))
    except Exception as exc:
        raise GGUFAnalysisError(f"failed to read {model_path}: {exc}") from exc

    fields = getattr(reader, "fields", {}) or {}
    architecture = _as_text(_field_value(fields, "general.architecture"))
    name = _as_text(_field_value(fields, "general.name"))

    n_layers = None
    context_length = None
    embedding_length = None
    n_heads = None
    n_kv_heads = None
    head_dim = None
    if architecture:
        n_layers = _as_int(_field_value(fields, f"{architecture}.block_count"))
        context_length = _as_int(_field_value(fields, f"{architecture}.context_length"))
        embedding_length = _as_int(_field_value(fields, f"{architecture}.embedding_length"))

        n_heads = _field_first_int(fields, f"{architecture}.attention.head_count")
        n_kv_heads = _field_first_int(fields, f"{architecture}.attention.head_count_kv")
        head_dim = _field_first_int(fields, f"{architecture}.attention.key_length")
        if head_dim is None and n_heads and embedding_length:
            head_dim = embedding_length // n_heads

    tensor_map: list[dict[str, Any]] = []
    layer_sizes: dict[int | None, int] = defaultdict(int)
    category_sizes: dict[str, int] = defaultdict(int)
    quant_sizes: dict[str, int] = defaultdict(int)
    quant_counts: dict[str, int] = defaultdict(int)
    warnings: list[dict[str, str]] = []
    warning_count = 0
    max_layer = -1
    # MoE weights split in two classes because llama.cpp's --n-cpu-moe only
    # moves routed experts (*_exps). Shared experts (*_shexp) and the router
    # (ffn_gate_inp) run for every token and must stay on the GPU with the
    # attention block, so folding them into expert_bytes would over-offload.
    expert_bytes = 0
    shared_expert_bytes = 0
    expert_layer_bytes: dict[int, int] = defaultdict(int)
    # What a tensor-type split needs (llama.cpp's --n-cpu-ffn): the bytes of a
    # layer that always ride with attention (its attn block and norms, and the
    # router/shared-expert tensors a MoE keeps active) versus the dense FFN
    # block that --n-cpu-ffn can move to the CPU on its own. The flag's regex is
    # `blk\.N\.ffn_(up|down|gate)\.`, so a shared expert (`ffn_gate_shexp`) is
    # NOT movable and must not be counted as dense FFN here.
    attn_layer_bytes: dict[int, int] = defaultdict(int)
    ffn_layer_bytes: dict[int, int] = defaultdict(int)
    fixed_bytes = 0

    for tensor in getattr(reader, "tensors", []):
        tensor_name = str(tensor.name)
        quant = _quant_name(tensor.tensor_type)
        category = _classify_tensor(tensor_name)
        layer = _layer_index(tensor_name)
        size = int(getattr(tensor, "n_bytes", 0) or 0)
        shape = [int(x) for x in getattr(tensor, "shape", [])]
        elements = int(getattr(tensor, "n_elements", 0) or 0)

        tensor_map.append(
            {
                "name": tensor_name,
                "shape": shape,
                "quant": quant,
                "category": category,
                "layer": layer,
                "n_elements": elements,
                "n_bytes": size,
            }
        )
        layer_sizes[layer] += size
        category_sizes[category] += size
        quant_sizes[quant] += size
        quant_counts[quant] += 1
        if "_exps" in tensor_name:
            expert_bytes += size
            if layer is not None:
                expert_layer_bytes[layer] += size
        elif "_shexp" in tensor_name:
            shared_expert_bytes += size
        if layer is None:
            # token_embd, output.weight and friends: resident either way, so
            # they are charged once instead of to every layer.
            fixed_bytes += size
        elif category == "ffn" and "_shexp" not in tensor_name:
            ffn_layer_bytes[layer] += size
        else:
            attn_layer_bytes[layer] += size
        if layer is not None:
            max_layer = max(max_layer, layer)

        rank = _quant_rank(quant)
        if (
            any(key in tensor_name for key in _CRITICAL_TENSORS)
            and rank is not None
            and rank < _CRITICAL_RANK
        ):
            warning_count += 1
            if len(warnings) < _MAX_WARNINGS:
                warnings.append(
                    {
                        "tensor": tensor_name,
                        "quant": quant,
                        "category": category,
                        "message": (
                            f"{tensor_name} is {quant}, below {_CRITICAL_THRESHOLD}; "
                            "this tensor is sensitive to quantization."
                        ),
                    }
                )

    if n_layers is None:
        n_layers = max_layer + 1
    # Both sources are untrusted: `block_count` is read from the header and the
    # fallback from tensor names, and either sizes the per-layer list below.
    n_layers = max(0, min(n_layers, _MAX_LAYERS))

    expert_layer_sizes = [int(expert_layer_bytes.get(i, 0)) for i in range(n_layers)]
    ffn_layer_sizes = [int(ffn_layer_bytes.get(i, 0)) for i in range(n_layers)]

    attention = _attention_profile([t["name"] for t in tensor_map], n_layers)

    filename_quant = None
    match = _QUANT_RE.search(model_path.name)
    if match:
        filename_quant = match.group(1).upper()

    dominant_quant = max(quant_sizes, key=lambda key: quant_sizes[key]) if quant_sizes else None

    return {
        "path": str(model_path),
        "metadata": {
            "architecture": architecture,
            "name": name,
            "n_layers": n_layers,
            "context_length": context_length,
            "embedding_length": embedding_length,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "n_params": _as_int(_field_value(fields, "general.parameter_count")),
            "quant": filename_quant or dominant_quant or "?",
            "dominant_quant": dominant_quant,
            "tensor_count": len(tensor_map),
            "file_size_bytes": os.path.getsize(model_path),
            # Routed experts are the only tensors --n-cpu-moe can move. The
            # per-layer list uses the model's layer indices so the planner can
            # fit the exact bytes of the layers that stay on the GPU.
            "expert_bytes": expert_bytes,
            "shared_expert_bytes": shared_expert_bytes,
            "expert_layer_bytes": expert_layer_sizes,
            "expert_layers": sum(1 for value in expert_layer_sizes if value > 0),
            # Split by tensor type for --n-cpu-ffn: the attention side (every
            # attn_* tensor plus its norms) always rides with the layer, only
            # the dense FFN block can be moved on its own, one layer at a time.
            # `fixed_bytes` are the tensors outside every block.
            "attn_bytes": sum(attn_layer_bytes.values()),
            "ffn_layer_bytes": ffn_layer_sizes,
            "fixed_bytes": fixed_bytes,
            **attention,
        },
        "tensor_map": tensor_map,
        "layer_distribution": dict(
            sorted(layer_sizes.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
        ),
        "category_sizes": dict(category_sizes),
        "quant_sizes": dict(quant_sizes),
        "quant_counts": dict(quant_counts),
        "warnings": warnings,
        # True when low-quant critical tensors were dropped by the cap, so a
        # consumer does not read a truncated list as the whole story.
        "warnings_truncated": warning_count > len(warnings),
    }


# Backwards-compatible alias kept for callers that used the original name.
analyze_gguf_structure = analyze_gguf
