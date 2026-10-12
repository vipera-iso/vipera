"""Write a tiny but structurally valid GGUF file for tests.

Only the header, metadata and tensor descriptors are written — the reader under
test never touches tensor data, so the fixture stays a few hundred bytes.
"""

from __future__ import annotations

import struct

F32 = 0  # GGML_TYPE_F32
UINT32 = 4
STRING = 8
ARRAY = 9


def _s(buf: bytearray, text: str) -> None:
    raw = text.encode("utf-8")
    buf += struct.pack("<Q", len(raw))
    buf += raw


def _kv_u32(buf: bytearray, key: str, value: int) -> None:
    _s(buf, key)
    buf += struct.pack("<I", UINT32)
    buf += struct.pack("<I", value)


def _kv_str(buf: bytearray, key: str, value: str) -> None:
    _s(buf, key)
    buf += struct.pack("<I", STRING)
    _s(buf, value)


def _kv_str_array(buf: bytearray, key: str, values: list[str]) -> None:
    _s(buf, key)
    buf += struct.pack("<I", ARRAY)
    buf += struct.pack("<I", STRING)
    buf += struct.pack("<Q", len(values))
    for v in values:
        _s(buf, v)


def _tensor(buf: bytearray, name: str, shape: tuple[int, ...], offset: int = 0) -> None:
    _s(buf, name)
    buf += struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<Q", d)
    buf += struct.pack("<I", F32)
    buf += struct.pack("<Q", offset)


def make_gguf(path: str, *, block_count: int = 2, n_experts: int = 4) -> None:
    """A 2-layer MoE model: 1 dense attn, 1 router, 1 shared expert, 1 routed expert per layer."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)

    tensors = [("token_embd.weight", (256, 8))]
    for i in range(block_count):
        tensors += [
            (f"blk.{i}.attn_q.weight", (256, 8)),
            (f"blk.{i}.ffn_gate_inp.weight", (256, 8)),
            (f"blk.{i}.ffn_gate_shexp.weight", (256, 64)),
            (f"blk.{i}.ffn_gate_exps.weight", (256, 64, n_experts)),
        ]

    kvs = 3
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", kvs)

    _kv_str(buf, "general.architecture", "llama")
    _kv_u32(buf, "llama.block_count", block_count)
    _kv_str_array(buf, "general.tags", ["test", "moe"])

    for name, shape in tensors:
        _tensor(buf, name, shape)

    with open(path, "wb") as f:
        f.write(buf)


def expected_bytes(*, block_count: int = 2, n_experts: int = 4) -> dict:
    """Byte totals the fixture must produce, derived from the same shapes above."""
    expert_per_layer = 256 * 64 * n_experts * 4  # F32
    return {
        "other": 256 * 8 * 4,  # token_embd
        "resident": block_count * 256 * 8 * 4,  # attn_q per layer
        "router": block_count * 256 * 8 * 4,  # ffn_gate_inp per layer
        "shared_expert": block_count * 256 * 64 * 4,  # ffn_gate_shexp per layer
        "expert": block_count * expert_per_layer,
    }
