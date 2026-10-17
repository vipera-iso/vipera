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
    r"(IQ\d+_\w+|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|BF16|F16|F32|Q\d+)",
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
}

_CRITICAL_TENSORS = ("ffn_down", "attn_v")
_CRITICAL_THRESHOLD = "Q4_K"


class GGUFAnalysisError(RuntimeError):
    """Raised when a GGUF file cannot be inspected."""


def _quant_name(tensor_type: Any) -> str:
    try:
        value = int(tensor_type)
    except (TypeError, ValueError):
        return str(tensor_type)
    return _QUANT_NAMES.get(value, f"TYPE_{value}")


def _quant_rank(name: str) -> float:
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
    return 0.0


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


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def analyze_gguf(path: str) -> dict[str, Any]:
    """Return metadata, tensor map, and size breakdown for a GGUF file.

    The return value is JSON-friendly except for integer layer keys, which are
    kept as integers for convenient sorting. It raises ``GGUFAnalysisError``
    with an actionable message when the file or the optional ``gguf`` package
    is unavailable.
    """
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
    if architecture:
        n_layers = _as_int(_field_value(fields, f"{architecture}.block_count"))
        context_length = _as_int(_field_value(fields, f"{architecture}.context_length"))
        embedding_length = _as_int(_field_value(fields, f"{architecture}.embedding_length"))

    tensor_map: list[dict[str, Any]] = []
    layer_sizes: dict[int | None, int] = defaultdict(int)
    category_sizes: dict[str, int] = defaultdict(int)
    quant_sizes: dict[str, int] = defaultdict(int)
    quant_counts: dict[str, int] = defaultdict(int)
    warnings: list[dict[str, str]] = []
    max_layer = -1

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
        if layer is not None:
            max_layer = max(max_layer, layer)

        if any(key in tensor_name for key in _CRITICAL_TENSORS) and _quant_rank(
            quant
        ) < _quant_rank(_CRITICAL_THRESHOLD):
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
            "quant": filename_quant or dominant_quant or "?",
            "dominant_quant": dominant_quant,
            "tensor_count": len(tensor_map),
            "file_size_bytes": os.path.getsize(model_path),
        },
        "tensor_map": tensor_map,
        "layer_distribution": dict(
            sorted(layer_sizes.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
        ),
        "category_sizes": dict(category_sizes),
        "quant_sizes": dict(quant_sizes),
        "quant_counts": dict(quant_counts),
        "warnings": warnings,
    }


# Backwards-compatible alias kept for callers that used the original name.
analyze_gguf_structure = analyze_gguf
