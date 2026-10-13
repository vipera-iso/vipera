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
# Metadata arrays are flat in every real GGUF; a corrupt count must not recurse until
# Python's own limit turns it into a RecursionError. Generous, and still a hard stop.
MAX_ARRAY_DEPTH = 32


class GGUFError(ValueError):
    """Raised when a file is not a well-formed GGUF."""


class UnknownQuantType(GGUFError):
    """Raised when a tensor uses a ggml quant type vipera has no block size for.

    Guessing a density here (fp16) silently produced byte totals several times
    too large, so this is loud instead.
    """

    def __init__(self, ggml_type: int, tensor_name: str | None = None):
        self.ggml_type = ggml_type
        self.tensor_name = tensor_name
        where = f" for tensor {tensor_name!r}" if tensor_name else ""
        super().__init__(
            f"unknown ggml quant type {ggml_type}{where}: vipera has no block size for it, "
            "so it cannot size the tensor. Add the type to _BLOCK in vipera/gguf.py "
            "(block sizes are in ggml-common.h)."
        )


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


def _remaining(f: BinaryIO, size: int) -> int:
    """Bytes left in a `size`-byte file at the current position."""
    return size - f.tell()


def _read_exact(f: BinaryIO, n: int, size: int) -> bytes:
    """Read exactly `n` bytes, bounded by the file's own size.

    Every length in a GGUF header is corruption-controlled data. A string length used
    to flow straight into `f.read(n)`, so 8 bytes claiming a 2^62-byte string made
    CPython attempt the allocation and raise MemoryError. Checking the bound *before*
    the read turns that into the same clean GGUFError as any other short header.
    """
    left = _remaining(f, size)
    if n > left:
        raise GGUFError(
            f"unexpected EOF at byte {f.tell()}: wanted {n} bytes, only {left} left "
            f"of a {size}-byte file"
        )
    data = f.read(n)
    if len(data) != n:
        raise GGUFError(f"unexpected EOF: wanted {n} bytes, got {len(data)}")
    return data


def _read_u32(f: BinaryIO, size: int) -> int:
    return struct.unpack("<I", _read_exact(f, 4, size))[0]


def _read_u64(f: BinaryIO, size: int) -> int:
    return struct.unpack("<Q", _read_exact(f, 8, size))[0]


def _read_string(f: BinaryIO, size: int) -> str:
    n = _read_u64(f, size)
    return _read_exact(f, n, size).decode("utf-8", errors="replace")


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


def _read_value(f: BinaryIO, vtype: int, size: int, depth: int = 0) -> Any:
    if vtype == ValueType.BOOL:
        return struct.unpack("<B", _read_exact(f, 1, size))[0] != 0
    if vtype == ValueType.STRING:
        return _read_string(f, size)
    if vtype == ValueType.ARRAY:
        elem_type = _read_u32(f, size)
        count = _read_u64(f, size)
        # Arrays of tensors are not a thing in GGUF, but nested arrays are not either:
        # the old reader recursed on the declared element type until Python's recursion
        # limit turned a corrupt count into a RecursionError.
        if depth >= MAX_ARRAY_DEPTH:
            raise GGUFError(
                f"array nested more than {MAX_ARRAY_DEPTH} deep at byte {f.tell()}: "
                "GGUF does not nest arrays, so this is a corrupt element type"
            )
        left = _remaining(f, size)
        if count > left:
            raise GGUFError(
                f"array of {count} elements at byte {f.tell()} cannot fit in the "
                f"{left} bytes left of a {size}-byte file"
            )
        return [_read_value(f, elem_type, size, depth + 1) for _ in range(count)]
    spec = _SCALAR.get(vtype)
    if spec is None:
        raise GGUFError(f"unknown GGUF value type {vtype}")
    fmt, width = spec
    return struct.unpack(fmt, _read_exact(f, width, size))[0]


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
            # Refuse to size an unknown/newer quant: the old fp16 fallback made
            # the byte totals silently wrong (3.76x too big on an MXFP4 file).
            raise UnknownQuantType(self.ggml_type, self.name)
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
    # Byte offset where tensor data starts (descriptors padded to the alignment).
    data_offset: int = 0

    def validate_extents(self) -> None:
        """Check every tensor's data lies inside the file.

        A descriptor-only file — one with no data section, so `file_size <=
        data_offset` — is allowed (test fixtures are built that way). A file that
        does carry a data region must satisfy `data_offset + offset + n_bytes <=
        file_size`; otherwise a half-truncated model still analyzed to a clean
        exit with wrong totals.
        """
        if self.file_size <= self.data_offset:
            return
        for t in self.tensors:
            end = self.data_offset + t.offset + t.n_bytes
            if end > self.file_size:
                raise GGUFError(
                    f"tensor {t.name!r} needs bytes up to {end}, past the "
                    f"{self.file_size}-byte end of file: the file is truncated "
                    "or its tensor offsets are wrong"
                )

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
        magic = _read_exact(f, 4, file_size)
        if magic != GGUF_MAGIC:
            raise GGUFError(f"not a GGUF file: magic is {magic!r}")
        version = _read_u32(f, file_size)
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = _read_u64(f, file_size)
        kv_count = _read_u64(f, file_size)

        # Neither list can hold more entries than the file has bytes left. Rejecting a
        # corrupt count up front means a 2^62 walks nothing instead of the whole file.
        left = _remaining(f, file_size)
        if kv_count > left or tensor_count > left:
            raise GGUFError(
                f"header claims {kv_count} metadata entries and {tensor_count} tensors, "
                f"which cannot fit in the {left} bytes left of a {file_size}-byte file"
            )

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f, file_size)
            vtype = _read_u32(f, file_size)
            metadata[key] = _read_value(f, vtype, file_size)

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = _read_string(f, file_size)
            n_dims = _read_u32(f, file_size)
            if n_dims > _remaining(f, file_size):
                raise GGUFError(
                    f"tensor {name!r} claims {n_dims} dimensions, more than the "
                    f"{_remaining(f, file_size)} bytes left of the file can hold"
                )
            dims = tuple(_read_u64(f, file_size) for _ in range(n_dims))
            ggml_type = _read_u32(f, file_size)
            offset = _read_u64(f, file_size)
            tensors.append(TensorInfo(name, dims, ggml_type, offset))

        # Tensor data starts at the next alignment boundary after the descriptors.
        alignment = metadata.get("general.alignment")
        if not isinstance(alignment, int) or alignment <= 0:
            alignment = DEFAULT_ALIGNMENT
        pos = f.tell()
        data_offset = (pos + alignment - 1) // alignment * alignment

    # A quant type we cannot size makes every byte total wrong: reject it up front
    # rather than returning a plausible-looking number.
    for t in tensors:
        if t.ggml_type not in _BLOCK:
            raise UnknownQuantType(t.ggml_type, t.name)

    gguf = GGUF(version=version, metadata=metadata, tensors=tensors,
                file_size=file_size, data_offset=data_offset)
    gguf.validate_extents()
    return gguf
