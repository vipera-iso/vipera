"""Minimal GGUF v3 reader.

Reads only the header, metadata and tensor descriptors — never the tensor data —
so it stays cheap on a weak device and can analyze a 20 GB file in milliseconds.

Format (little-endian throughout):
    magic        u8[4]  == b"GGUF"
    version      u32
    tensor_count u64
    kv_count     u64
    metadata KVs (kv_count times)
    tensor infos (tensor_count times)
    padding to `general.alignment` (default 32)
    tensor data
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32


class GGUFError(ValueError):
    """Raised when a file is not a well-formed GGUF."""


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    Q4_0_4_4 = 31
    Q4_0_4_8 = 32
    Q4_0_8_8 = 33
    TQ1_0 = 34
    TQ2_0 = 35


# ggml_type -> (weights per block, bytes per block)
_BLOCK: dict[int, tuple[int, int]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.BF16: (1, 2),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q8_1: (32, 36),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
    GGMLType.Q8_K: (256, 292),
    GGMLType.IQ2_XXS: (256, 66),
    GGMLType.IQ2_XS: (256, 74),
    GGMLType.IQ3_XXS: (256, 98),
    GGMLType.IQ1_S: (256, 50),
    GGMLType.IQ4_NL: (32, 18),
    GGMLType.IQ3_S: (256, 110),
    GGMLType.IQ2_S: (256, 82),
    GGMLType.IQ4_XS: (256, 136),
    GGMLType.IQ1_M: (256, 56),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (1, 2),
    GGMLType.I32: (1, 4),
    GGMLType.I64: (1, 8),
    GGMLType.F64: (1, 8),
    GGMLType.Q4_0_4_4: (32, 18),
    GGMLType.Q4_0_4_8: (32, 18),
    GGMLType.Q4_0_8_8: (32, 18),
    GGMLType.TQ1_0: (256, 54),
    GGMLType.TQ2_0: (256, 66),
}


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFError(f"unexpected EOF: wanted {n} bytes, got {len(data)}")
    return data


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f: BinaryIO) -> str:
    n = _read_u64(f)
    return _read_exact(f, n).decode("utf-8", errors="replace")


_SCALAR = {
    ValueType.UINT8: ("<B", 1),
    ValueType.INT8: ("<b", 1),
    ValueType.UINT16: ("<H", 2),
    ValueType.INT16: ("<h", 2),
    ValueType.UINT32: ("<I", 4),
    ValueType.INT32: ("<i", 4),
    ValueType.FLOAT32: ("<f", 4),
    ValueType.UINT64: ("<Q", 8),
    ValueType.INT64: ("<q", 8),
    ValueType.FLOAT64: ("<d", 8),
}


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype == ValueType.BOOL:
        return struct.unpack("<B", _read_exact(f, 1))[0] != 0
    if vtype == ValueType.STRING:
        return _read_string(f)
    if vtype == ValueType.ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        # Arrays of tensors are not a thing in GGUF, but nested arrays are not either.
        return [_read_value(f, elem_type) for _ in range(count)]
    spec = _SCALAR.get(vtype)
    if spec is None:
        raise GGUFError(f"unknown GGUF value type {vtype}")
    fmt, size = spec
    return struct.unpack(fmt, _read_exact(f, size))[0]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        block = _BLOCK.get(self.ggml_type)
        if block is None:
            # Unknown/newer quant: assume fp16 density rather than silently lying.
            return self.n_elements * 2
        per_block, bytes_per_block = block
        if self.n_elements % per_block != 0:
            raise GGUFError(
                f"tensor {self.name!r} has {self.n_elements} elements, not a multiple of "
                f"the {per_block}-element block for the quant type"
            )
        return (self.n_elements // per_block) * bytes_per_block

    @property
    def type_name(self) -> str:
        try:
            return GGMLType(self.ggml_type).name
        except ValueError:
            return f"UNKNOWN({self.ggml_type})"


@dataclass
class GGUF:
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    file_size: int

    @property
    def architecture(self) -> str | None:
        arch = self.metadata.get("general.architecture")
        return arch if isinstance(arch, str) else None

    @property
    def tensor_bytes(self) -> int:
        return sum(t.n_bytes for t in self.tensors)

    def tensor(self, name: str) -> TensorInfo | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


def read_gguf(path: str) -> GGUF:
    """Parse the header, metadata and tensor descriptors of a GGUF file."""
    import os

    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise GGUFError(f"not a GGUF file: magic is {magic!r}")
        version = _read_u32(f)
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = _read_u64(f)
        kv_count = _read_u64(f)

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            vtype = _read_u32(f)
            metadata[key] = _read_value(f, vtype)

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = _read_string(f)
            n_dims = _read_u32(f)
            dims = tuple(_read_u64(f) for _ in range(n_dims))
            ggml_type = _read_u32(f)
            offset = _read_u64(f)
            tensors.append(TensorInfo(name, dims, ggml_type, offset))

    return GGUF(version=version, metadata=metadata, tensors=tensors, file_size=file_size)
