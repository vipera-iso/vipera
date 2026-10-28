"""Write GGUF files — and edit which expert slabs live inside them.

`vipera/gguf.py` reads descriptors and never touches tensor data, so nothing in the
project could produce a model file; that is what blocked every format-level
experiment in `docs/ALGORITHMS.md` (pack 19 papers C and D). This is that writer.

`rewrite_experts` copies a model and rewrites only the **routed expert** tensors.
Their expert dimension is the slowest-varying one, so expert `e` of a
`(n_embd, n_ff, n_experts)` tensor is one contiguous run of bytes. Both operations
the literature asks for are therefore byte moves over already-quantized slabs, and
neither needs the quantizer:

* reorder — `experts[l]` is a permutation. This is the co-activation "flash layout"
  of Neuralink (arXiv:2410.19274): the experts a token actually uses are laid out
  next to each other, so a trace-driven access order is a sequential read.
* prune — `experts[l]` is a subset. This is the write side of `vipera prune`: the
  chosen slabs are kept and the tensor's expert dimension shrinks to match, so the
  pruned plan becomes a file other runtimes can load.

Everything else — every dense tensor, and the metadata block — is copied through
untouched. The metadata block is copied *verbatim* precisely so no GGUF value type
has to be re-encoded, which is also why the reader records where it ends.

    vipera rewrite --model models/big.gguf --out models/small.gguf \\
        --from-trace trace.jsonl --prune-to 16
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import BinaryIO, Iterable, Mapping, Sequence

from .analyze import _layer_of, is_expert_tensor
from .gguf import DEFAULT_ALIGNMENT, GGUF, GGUFError, TensorInfo, read_gguf

_HEADER_BYTES = 24  # magic(4) + version(4) + tensor_count(8) + kv_count(8)
_CHUNK = 4 << 20  # copy/resize unit


@dataclass
class RewriteReport:
    """What `rewrite_experts` did, in bytes and in experts."""

    src: str
    dest: str
    tensors: int = 0
    edited_tensors: int = 0
    per_layer: dict[int, int] = field(default_factory=dict)
    experts_before: int = 0
    experts_after: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    alignment: int = DEFAULT_ALIGNMENT
    data_offset: int = 0

    def as_dict(self) -> dict:
        return {
            "src": os.path.basename(self.src),
            "dest": os.path.basename(self.dest),
            "tensors": self.tensors,
            "edited_tensors": self.edited_tensors,
            "experts_before": self.experts_before,
            "experts_after": self.experts_after,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "delta_bytes": self.bytes_after - self.bytes_before,
            "alignment": self.alignment,
            "per_layer_experts": {str(k): v for k, v in sorted(self.per_layer.items())},
        }


def expert_slab_bytes(t: TensorInfo, n_experts: int) -> int:
    """Bytes of one expert inside a 3-D routed-expert tensor.

    Refuses when the slab is not a whole number of bytes or when a slab would cut
    a quantization block in half: both mean the tensor is not laid out as
    `(n_embd, n_ff, n_experts)` and moving slabs would corrupt it.
    """
    if len(t.shape) < 3:
        raise GGUFError(
            f"tensor {t.name!r} has shape {t.shape}; a routed-expert tensor needs "
            "at least 3 dimensions so its experts are contiguous"
        )
    if int(t.shape[2]) != n_experts:
        raise GGUFError(f"tensor {t.name!r} has {t.shape[2]} experts, expected {n_experts}")
    total = t.n_bytes
    if total % n_experts != 0:
        raise GGUFError(
            f"tensor {t.name!r} is {total} bytes, not divisible by {n_experts} experts"
        )
    return total // n_experts


def _validate_layer_experts(t: TensorInfo, ids: Sequence[int]) -> None:
    n_experts = int(t.shape[2]) if len(t.shape) >= 3 else 0
    if len(t.shape) < 3:
        raise GGUFError(
            f"tensor {t.name!r} has shape {t.shape}; cannot select experts from it"
        )
    if len(set(ids)) != len(ids):
        raise GGUFError(f"tensor {t.name!r}: expert list repeats an id: {list(ids)}")
    for e in ids:
        if not 0 <= e < n_experts:
            raise GGUFError(
                f"tensor {t.name!r}: expert id {e} is outside its {n_experts} experts"
            )
    if not ids:
        raise GGUFError(f"tensor {t.name!r}: refusing to write a tensor with no experts")


def _write_string(buf: bytearray, text: str) -> None:
    import struct

    raw = text.encode("utf-8")
    buf += struct.pack("<Q", len(raw))
    buf += raw


def _write_descriptor(buf: bytearray, name: str, shape: tuple[int, ...], ggml_type: int,
                      offset: int) -> None:
    import struct

    _write_string(buf, name)
    buf += struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<Q", d)
    buf += struct.pack("<I", ggml_type)
    buf += struct.pack("<Q", offset)


def _copy_range(src: BinaryIO, dst: BinaryIO, offset: int, length: int,
                slabs: Sequence[tuple[int, int]] | None = None) -> None:
    """Copy `length` bytes at `offset`, or only the given (start, size) slabs of it."""
    if length == 0:
        return
    if slabs is None:
        src.seek(offset)
        left = length
        while left > 0:
            n = min(_CHUNK, left)
            dst.write(src.read(n))
            left -= n
        return
    for start, size in slabs:
        src.seek(offset + start)
        left = size
        while left > 0:
            n = min(_CHUNK, left)
            dst.write(src.read(n))
            left -= n


def rewrite_experts(src: str, dest: str, experts: Mapping[int, Sequence[int]], *,
                    alignment: int = DEFAULT_ALIGNMENT) -> RewriteReport:
    """Copy `src` to `dest`, rewriting the routed-expert tensors of each layer in `experts`.

    `experts` maps a block index to the expert ids to write **in file order**: a
    permutation reorders the slabs, a subset prunes them. Every other tensor is
    copied byte for byte.
    """
    if alignment <= 0 or alignment & (alignment - 1):
        raise GGUFError(f"alignment must be a positive power of two, got {alignment}")

    g = read_gguf(src)
    if g.kv_end < _HEADER_BYTES:
        raise GGUFError(
            f"{src}: reader did not report the end of the metadata block; "
            "cannot rewrite without re-encoding it"
        )

    report = RewriteReport(src=src, dest=dest, tensors=len(g.tensors),
                           bytes_before=g.file_size, alignment=alignment)

    # Work out what each tensor looks like on the way out before opening the file:
    # the descriptor block has to be written in one pass, ahead of the data.
    plans: list[tuple[TensorInfo, tuple[int, ...], Sequence[tuple[int, int]] | None, int]] = []
    offset = 0
    for t in g.tensors:
        layer = _layer_of(t.name)
        wanted = experts.get(layer) if (layer is not None and is_expert_tensor(t.name)) else None

        if wanted is None:
            out_shape = t.shape
            slabs = None
            n_bytes = t.n_bytes
        else:
            _validate_layer_experts(t, wanted)
            slab = expert_slab_bytes(t, int(t.shape[2]))
            out_shape = (int(t.shape[0]), int(t.shape[1]), len(wanted))
            slabs = [(e * slab, slab) for e in wanted]
            n_bytes = slab * len(wanted)
            report.edited_tensors += 1
            report.experts_before += int(t.shape[2])
            report.experts_after += len(wanted)
            report.per_layer[layer] = len(wanted)

        offset = (offset + alignment - 1) // alignment * alignment
        plans.append((t, out_shape, slabs, offset))
        offset += n_bytes

    data_bytes = offset
    requested = {li for li in experts}
    found = set(report.per_layer)
    missing = sorted(requested - found)
    if missing:
        raise GGUFError(
            f"{src}: no routed-expert tensor found for layer(s) {missing}; "
            f"available layers: {sorted({_layer_of(t.name) for t in g.tensors if _layer_of(t.name) is not None and is_expert_tensor(t.name)})}"
        )

    with open(src, "rb") as fin, open(dest, "wb") as fout:
        header = fin.read(_HEADER_BYTES)
        if len(header) != _HEADER_BYTES:
            raise GGUFError(f"{src}: file is too short to hold a GGUF header")
        fout.write(header)

        # metadata block, verbatim: this is what keeps every value type intact
        fin.seek(_HEADER_BYTES)
        left = g.kv_end - _HEADER_BYTES
        while left > 0:
            n = min(_CHUNK, left)
            fout.write(fin.read(n))
            left -= n

        descriptors = bytearray()
        for t, out_shape, _, out_offset in plans:
            _write_descriptor(descriptors, t.name, out_shape, t.ggml_type, out_offset)
        fout.write(descriptors)

        pos = _HEADER_BYTES + (g.kv_end - _HEADER_BYTES) + len(descriptors)
        pad = (-pos) % alignment
        if pad:
            fout.write(b"\x00" * pad)
        data_offset = pos + pad

        written = 0
        for t, _, slabs, out_offset in plans:
            if written < out_offset:
                fout.write(b"\x00" * (out_offset - written))
                written = out_offset
            _copy_range(fin, fout, g.data_offset + t.offset, t.n_bytes, slabs)
            written += (t.n_bytes if slabs is None else sum(size for _, size in slabs))

    report.bytes_after = os.path.getsize(dest)
    report.data_offset = data_offset
    return report


def expert_order(routing: Iterable[Iterable[Iterable[int]]], *, n_experts: int,
                 prune_to: int | None = None) -> dict[int, list[int]]:
    """Per-layer expert ids ranked by how often the trace routes to them.

    All experts are returned unless `prune_to` caps the list, so the same call
    describes a pure reorder (co-activation layout) and a prune. Ties break on the
    expert id, so the order is deterministic.
    """
    if n_experts <= 0:
        raise ValueError(f"n_experts must be positive, got {n_experts}")
    if prune_to is not None and not 0 < prune_to <= n_experts:
        raise ValueError(f"prune_to must be in (0, {n_experts}], got {prune_to}")

    counts: dict[int, list[int]] = {}
    for token in routing:
        for li, experts in enumerate(token):
            layer = counts.setdefault(li, [0] * n_experts)
            for e in experts:
                if not 0 <= e < n_experts:
                    raise ValueError(f"trace routes to expert {e}, outside 0..{n_experts - 1}")
                layer[e] += 1

    order: dict[int, list[int]] = {}
    for li, freq in counts.items():
        ranked = sorted(range(n_experts), key=lambda e: (-freq[e], e))
        order[li] = ranked[:prune_to] if prune_to is not None else ranked
    if not order:
        raise ValueError("trace is empty")
    return order


__all__ = ["RewriteReport", "rewrite_experts", "expert_order", "expert_slab_bytes"]
