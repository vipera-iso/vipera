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


def _tensor(buf: bytearray, name: str, shape: tuple[int, ...], offset: int = 0,
            ggml_type: int = F32) -> None:
    _s(buf, name)
    buf += struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<Q", d)
    buf += struct.pack("<I", ggml_type)
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


def make_imatrix_gguf(path: str, *, tensors: dict[str, list[float]], shape: tuple[int, int] | None = None) -> None:
    """An imatrix-shaped GGUF: F32 tensors only, with a real data section.

    `tensors` maps a tensor name to its flat little-endian F32 payload; `shape` applies
    to every tensor (the probe reads 2-D expert matrices), defaulting to 1-D.
    """
    names = list(tensors)
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", len(names))
    buf += struct.pack("<Q", 2)
    _kv_str(buf, "general.architecture", "llama")
    _kv_str(buf, "general.name", "synthetic-imatrix")
    offset = 0
    for name in names:
        dims = shape if shape is not None else (len(tensors[name]),)
        _tensor(buf, name, dims, offset=offset)
        offset += len(tensors[name]) * 4
    buf += b"\x00" * ((-len(buf)) % 32)
    for name in names:
        buf += struct.pack(f"<{len(tensors[name])}f", *tensors[name])
    with open(path, "wb") as f:
        f.write(buf)


def make_dense_gguf(path: str, *, block_count: int = 2) -> None:
    """A dense (non-MoE) model: no routed-expert tensor anywhere, so the expert
    rewrite path has nothing to act on."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 1 + 3 * block_count)  # tensor_count
    buf += struct.pack("<Q", 2)
    _kv_str(buf, "general.architecture", "llama")
    _kv_u32(buf, "llama.block_count", block_count)
    _tensor(buf, "token_embd.weight", (256, 8))
    for i in range(block_count):
        _tensor(buf, f"blk.{i}.attn_q.weight", (256, 8))
        _tensor(buf, f"blk.{i}.ffn_gate.weight", (256, 64))
        _tensor(buf, f"blk.{i}.ffn_down.weight", (64, 256))
    with open(path, "wb") as f:
        f.write(buf)


def make_unknown_quant_gguf(path: str, ggml_type: int = 999) -> None:
    """A structurally valid header whose single tensor uses a quant type vipera cannot size.

    Descriptor-only (no tensor data), like the main fixture, so it isolates the
    unknown-type check from the extent check.
    """
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 1)  # tensor_count
    buf += struct.pack("<Q", 1)  # kv_count
    _kv_str(buf, "general.architecture", "llama")
    _tensor(buf, "blk.0.ffn_gate_exps.weight", (256, 64, 4), ggml_type=ggml_type)
    with open(path, "wb") as f:
        f.write(buf)


def make_gguf_with_data(path: str, *, data_bytes: int, tensor_elements: int = 2048) -> int:
    """An F32 GGUF with a real (but possibly short) data section for extent tests.

    Writes `data_bytes` bytes of tensor data after the alignment padding. The single
    tensor claims `tensor_elements` F32 values, so `data_bytes` smaller than
    `tensor_elements * 4` models a half-truncated file. Returns the file size.
    """
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 1)  # tensor_count
    buf += struct.pack("<Q", 1)  # kv_count
    _kv_str(buf, "general.architecture", "llama")
    _tensor(buf, "blk.0.attn_q.weight", (tensor_elements,), offset=0)
    pad = (-len(buf)) % 32  # readers assume data starts on a 32-byte boundary
    buf += b"\x00" * pad
    buf += b"\x00" * data_bytes
    with open(path, "wb") as f:
        f.write(buf)
    return len(buf)


# The malformed-header fixtures below are deliberately written by hand rather than with
# `_kv_*`: the whole point is a length or count that no well-formed helper would emit.


def _header(buf: bytearray, *, kv_count: int, tensor_count: int) -> None:
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", tensor_count)
    buf += struct.pack("<Q", kv_count)


def make_gguf_with_corrupt_string_length(path: str, *, length: int = 2 ** 62) -> None:
    """A header whose single metadata string claims `length` bytes of text.

    Nothing like that much data is in the file, and the reader must reject the length
    *before* allocating for it — this is the input that used to raise MemoryError.
    """
    buf = bytearray()
    _header(buf, kv_count=1, tensor_count=0)
    _s(buf, "k")
    buf += struct.pack("<I", STRING)
    buf += struct.pack("<Q", length)
    with open(path, "wb") as f:
        f.write(buf)


def make_gguf_with_nested_arrays(path: str, *, depth: int = 3000) -> None:
    """A metadata value that is an array of arrays, `depth` levels deep.

    GGUF metadata arrays are flat; a deep chain means a corrupt element type, and the
    reader used to follow it until Python raised RecursionError.
    """
    buf = bytearray()
    _header(buf, kv_count=1, tensor_count=0)
    _s(buf, "k")
    buf += struct.pack("<I", ARRAY)
    for _ in range(depth):
        buf += struct.pack("<I", ARRAY)
        buf += struct.pack("<Q", 1)
    buf += struct.pack("<I", UINT32)
    buf += struct.pack("<Q", 0)
    with open(path, "wb") as f:
        f.write(buf)


def make_gguf_with_oversized_array(path: str, *, count: int = 2 ** 62) -> None:
    """A flat metadata array claiming more elements than the file could hold."""
    buf = bytearray()
    _header(buf, kv_count=1, tensor_count=0)
    _s(buf, "k")
    buf += struct.pack("<I", ARRAY)
    buf += struct.pack("<I", UINT32)
    buf += struct.pack("<Q", count)
    with open(path, "wb") as f:
        f.write(buf)


def make_gguf_with_bogus_counts(path: str, *, kv_count: int = 0,
                                tensor_count: int = 0) -> None:
    """A header whose metadata/tensor counts exceed what the file could hold."""
    buf = bytearray()
    _header(buf, kv_count=kv_count, tensor_count=tensor_count)
    with open(path, "wb") as f:
        f.write(buf)


def make_gguf_with_expert_data(path: str, *, block_count: int = 2, n_experts: int = 4,
                               alignment: int = 32, n_embd: int = 64, n_ff: int = 64) -> dict:
    """A GGUF *with* a data section, where every tensor is filled with a recognizable pattern.

    Dense tensors are filled with a per-tensor byte, and each expert slab of a routed
    expert tensor with a byte that encodes (layer, expert), so a reorder or a prune can
    be checked byte for byte rather than by size alone. Returns the byte value used for
    each (layer, expert) pair.
    """
    tensors = [("token_embd.weight", (n_embd, 8))]
    for i in range(block_count):
        tensors += [
            (f"blk.{i}.attn_q.weight", (n_embd, 8)),
            (f"blk.{i}.ffn_gate_inp.weight", (n_embd, 8)),
            (f"blk.{i}.ffn_gate_shexp.weight", (n_embd, n_ff)),
            (f"blk.{i}.ffn_gate_exps.weight", (n_embd, n_ff, n_experts)),
        ]

    patterns: dict[tuple[int, int], int] = {}
    offsets: list[int] = []
    offset = 0
    for name, shape in tensors:
        offset = (offset + alignment - 1) // alignment * alignment
        offsets.append(offset)
        n = 1
        for d in shape:
            n *= d
        offset += n * 4  # F32

    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", 3)
    _kv_str(buf, "general.architecture", "llama")
    _kv_u32(buf, "llama.block_count", block_count)
    _kv_u32(buf, "general.alignment", alignment)
    for (name, shape), off in zip(tensors, offsets):
        _tensor(buf, name, shape, offset=off)

    data = bytearray()
    for (name, shape), off in zip(tensors, offsets):
        pad = off - len(data)
        data += b"\x00" * pad
        n = 1
        for d in shape:
            n *= d
        if "_exps" in name:
            layer = _layer_index(name)
            slab = n // shape[2] * 4
            for e in range(shape[2]):
                value = (0x10 * (layer + 1) + e) & 0xFF
                patterns[(layer, e)] = value
                data += bytes([value]) * slab
        else:
            data += bytes([0xC0 + _layer_index(name)]) * (n * 4)

    pad = (-len(buf)) % alignment
    buf += b"\x00" * pad
    buf += data
    with open(path, "wb") as f:
        f.write(buf)
    return patterns


def _layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[1]) if len(parts) > 2 and parts[0] == "blk" else 0


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
