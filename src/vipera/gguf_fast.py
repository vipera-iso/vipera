"""Fast reader for the GGUF index (header, metadata, tensor table).

The `gguf` package builds a Python object per metadata item, which is correct but
slow on the tokenizer vocabulary: a 151k-token model costs 4.1 s to open because
every token string is decoded through a numpy memmap slice. `analyze_gguf` never
reads the vocabulary, so this module parses the same bytes with `mmap` +
`struct` and materializes only what the analyzer consumes:

* scalar metadata and small numeric arrays (attention head counts and friends),
* the tensor table (name, shape, type) from which sizes are derived.

Anything else is skipped by advancing a cursor, so a 6.9 GB file costs about the
same as its header. The object shapes match the `gguf` package closely enough
that `gguf_analyzer` runs unchanged on either reader:

    reader.fields[key].contents()   -> bytes | int | list[int]
    reader.tensors[i]               -> .name .shape .tensor_type .n_elements .n_bytes

Unknown or malformed files raise `GGUFReadError`; the caller falls back to
`gguf.GGUFReader`.
"""

from __future__ import annotations

import mmap
import struct
from typing import Any

_MAGIC = b"GGUF"

# GGUF metadata value types.
_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_INT64 = 11
_FLOAT64 = 12

_SCALAR_FMT = {
    _UINT8: "<B",
    _INT8: "<b",
    _UINT16: "<H",
    _INT16: "<h",
    _UINT32: "<I",
    _INT32: "<i",
    _FLOAT32: "<f",
    _BOOL: "<?",
    _UINT64: "<Q",
    _INT64: "<q",
    _FLOAT64: "<d",
}
_SCALAR_SIZE = {kind: struct.calcsize(fmt) for kind, fmt in _SCALAR_FMT.items()}
_UNPACK = {kind: struct.Struct(fmt).unpack_from for kind, fmt in _SCALAR_FMT.items()}

_U32 = struct.Struct("<I").unpack_from
_U64 = struct.Struct("<Q").unpack_from

# Arrays longer than this are never metadata the analyzer wants (the vocabulary
# and its merges are the only arrays this big), so they are skipped unread.
_MAX_ARRAY = 4096
# A string longer than this is a template or a licence, not a name.
_MAX_STRING = 4096

_SKIP = object()


class GGUFReadError(Exception):
    """The file is not a GGUF index this reader can parse."""


class _Field:
    """Mimics `gguf.ReaderField` for the one method the analyzer calls."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def contents(self) -> Any:
        return self._value


class _Tensor:
    """Mimics the `gguf.ReaderTensor` attributes the analyzer reads."""

    __slots__ = ("n_bytes", "n_elements", "name", "shape", "tensor_type")

    def __init__(
        self, name: str, shape: list[int], tensor_type: int, n_elements: int, n_bytes: int
    ) -> None:
        self.name = name
        self.shape = shape
        self.tensor_type = tensor_type
        self.n_elements = n_elements
        self.n_bytes = n_bytes


def _quant_sizes() -> dict[int, tuple[int, int]]:
    """`{type_id: (block_size, type_size)}`, taken from the gguf package.

    Reusing its table keeps `n_bytes` bit-identical to the reader this replaces;
    a local copy would be one more thing to drift. The import is lazy so a
    caller with the fast path only still pays nothing for numpy.
    """
    try:
        from gguf.constants import GGML_QUANT_SIZES
    except Exception as exc:  # pragma: no cover - gguf is a declared extra
        raise GGUFReadError(f"gguf not installed: {exc}") from exc
    return {int(ggml_type): (int(pair[0]), int(pair[1])) for ggml_type, pair in GGML_QUANT_SIZES.items()}


class _Reader:
    def __init__(self, buf: mmap.mmap) -> None:
        self.buf = buf
        self.size = len(buf)

    # -- primitives ---------------------------------------------------------

    def _need(self, pos: int, length: int, what: str) -> None:
        if length < 0 or pos < 0 or pos + length > self.size:
            raise GGUFReadError(f"truncated {what} at offset {pos} (need {length} bytes)")

    def _count(self, pos: int, wide: bool) -> tuple[int, int]:
        step = 8 if wide else 4
        self._need(pos, step, "count")
        value = (_U64 if wide else _U32)(self.buf, pos)[0]
        return value, pos + step

    def _string(self, pos: int, wide: bool, what: str = "string") -> tuple[str, int]:
        length, pos = self._count(pos, wide)
        self._need(pos, length, what)
        raw = bytes(self.buf[pos : pos + length])
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFReadError(f"{what} is not UTF-8: {exc}") from exc
        return text, pos + length

    # -- values -------------------------------------------------------------

    def _value(self, pos: int, vtype: int, wide: bool) -> tuple[Any, int]:
        buf = self.buf
        if vtype == _STRING:
            length, pos = self._count(pos, wide)
            self._need(pos, length, "string value")
            if length > _MAX_STRING:
                return _SKIP, pos + length
            try:
                return bytes(buf[pos : pos + length]).decode("utf-8"), pos + length
            except UnicodeDecodeError as exc:
                raise GGUFReadError(f"metadata string is not UTF-8: {exc}") from exc

        if vtype == _ARRAY:
            self._need(pos, 4, "array type")
            item = _U32(buf, pos)[0]
            count, pos = self._count(pos + 4, wide)

            if item == _STRING:
                # Skipped, but the end is only known by walking the lengths.
                # This is the one loop left that runs per vocabulary entry, so
                # it keeps no bounds check per item: the final `_need` covers
                # the whole walk, and a bad length lands past the file there.
                step = 8 if wide else 4
                unpack = _U64 if wide else _U32
                end = pos
                for _ in range(count):
                    end += step + unpack(buf, end)[0]
                self._need(end, 0, "array strings")
                return _SKIP, end

            unit = _SCALAR_SIZE.get(item)
            if unit is None:
                raise GGUFReadError(f"unsupported array item type {item}")
            total = unit * count
            self._need(pos, total, "array")
            if count > _MAX_ARRAY:
                return _SKIP, pos + total
            unpack = _UNPACK[item]
            return [unpack(buf, pos + i * unit)[0] for i in range(count)], pos + total

        unit = _SCALAR_SIZE.get(vtype)
        if unit is None:
            raise GGUFReadError(f"unsupported metadata value type {vtype}")
        self._need(pos, unit, "scalar")
        return _UNPACK[vtype](buf, pos)[0], pos + unit

    # -- whole index --------------------------------------------------------

    def read(self) -> FastGGUFReader:
        buf = self.buf
        if self.size < 24:
            raise GGUFReadError("file is too small to be GGUF")
        if bytes(buf[0:4]) != _MAGIC:
            raise GGUFReadError("not a GGUF file (bad magic)")
        self._need(4, 4, "version")
        version = _U32(buf, 4)[0]
        if version not in (1, 2, 3):
            raise GGUFReadError(f"unsupported GGUF version {version}")
        wide = version >= 2
        pos = 8
        tensor_count, pos = self._count(pos, wide)
        kv_count, pos = self._count(pos, wide)

        fields: dict[str, _Field] = {}
        for _ in range(kv_count):
            key, pos = self._string(pos, wide, "metadata key")
            self._need(pos, 4, "metadata type")
            vtype = _U32(buf, pos)[0]
            value, pos = self._value(pos + 4, vtype, wide)
            if value is not _SKIP:
                fields[key] = _Field(value)

        sizes = _quant_sizes()
        tensors: list[_Tensor] = []
        seen: set[str] = set()
        for _ in range(tensor_count):
            name, pos = self._string(pos, wide, "tensor name")
            if name in seen:
                raise GGUFReadError(f"duplicate tensor name {name}")
            seen.add(name)
            self._need(pos, 4, "tensor rank")
            n_dims = _U32(buf, pos)[0]
            pos += 4
            dims: list[int] = []
            dim_step = 8 if wide else 4
            for _ in range(n_dims):
                self._need(pos, dim_step, "tensor dimension")
                dims.append((_U64 if wide else _U32)(buf, pos)[0])
                pos += dim_step
            self._need(pos, 4 + 8, "tensor type/offset")
            tensor_type = _U32(buf, pos)[0]
            pos += 4 + 8  # type + u64 offset (offsets are 64-bit in every version)

            block_size, type_size = sizes.get(tensor_type, (0, 0))
            if block_size <= 0:
                raise GGUFReadError(f"unknown tensor type {tensor_type} on {name}")
            n_elements = 1
            for dim in dims:
                n_elements *= dim
            tensors.append(
                _Tensor(
                    name=name,
                    # File order, like gguf.ReaderTensor.shape: the analyzer
                    # reports it verbatim, so reversing it here would change
                    # every entry of tensor_map.
                    shape=list(dims),
                    tensor_type=tensor_type,
                    n_elements=n_elements,
                    n_bytes=n_elements * type_size // block_size,
                )
            )

        return FastGGUFReader(fields, tensors)


class FastGGUFReader:
    """Drop-in for the two attributes `gguf_analyzer` uses."""

    __slots__ = ("fields", "tensors")

    def __init__(self, fields: dict[str, _Field], tensors: list[_Tensor]) -> None:
        self.fields = fields
        self.tensors = tensors


def read_gguf(path: str) -> FastGGUFReader:
    """Parse `path` without decoding what the analyzer does not need."""
    import os

    try:
        fd = os.open(os.path.expanduser(path), os.O_RDONLY)
    except OSError as exc:
        raise GGUFReadError(f"cannot open {path}: {exc}") from exc
    try:
        buf = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    except (ValueError, OSError) as exc:
        raise GGUFReadError(f"cannot map {path}: {exc}") from exc
    finally:
        os.close(fd)  # the mapping holds its own reference
    try:
        return _Reader(buf).read()
    finally:
        buf.close()
