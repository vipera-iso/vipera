"""Verify a GGUF rewrite: did it preserve everything it was supposed to?

`vipera rewrite` claims two things - the tensors it did not touch come through byte
for byte, and the ones it did touch are in the order asked for. This checks the
first claim on the real file, because a fixture cannot prove a 10 GB copy is
lossless: it reads both files tensor by tensor and compares digests.

The strongest use is a **round trip**: permute the experts, then permute them back
with the inverse order. If the writer is lossless, every tensor in the result has
to hash identically to the original, and the two files have to disagree about
nothing at all.

    python3 -m bench.verify_rewrite --src models/big.gguf --dst roundtrip.gguf

Also reports *which* tensors differ, which is what a deliberate permutation looks
like: a reorder changes `*_exps.weight` and nothing else.

Exit codes: 0 every tensor identical, 1 a mismatch (printed), 2 usage/read failure.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

from vipera.gguf import GGUFError, read_gguf

_CHUNK = 4 << 20


def tensor_digest(path: str, data_offset: int, offset: int, n_bytes: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        f.seek(data_offset + offset)
        left = n_bytes
        while left > 0:
            block = f.read(min(_CHUNK, left))
            if not block:
                raise GGUFError(f"{os.path.basename(path)}: short read, {left} bytes missing")
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def compare(src: str, dst: str) -> tuple[list[str], int]:
    """Return (problems, tensors compared). Problems include the names that differ."""
    a, b = read_gguf(src), read_gguf(dst)
    problems: list[str] = []

    if a.metadata != b.metadata:
        keys = sorted(set(a.metadata) | set(b.metadata))
        differing = [k for k in keys if a.metadata.get(k) != b.metadata.get(k)]
        problems.append(f"metadata differs in {len(differing)} key(s): {differing[:5]}")

    if len(a.tensors) != len(b.tensors):
        problems.append(f"tensor count {len(a.tensors)} -> {len(b.tensors)}")
    else:
        for ta, tb in zip(a.tensors, b.tensors):
            if (ta.name, ta.shape, ta.ggml_type) != (tb.name, tb.shape, tb.ggml_type):
                problems.append(
                    f"descriptor changed: {ta.name} {ta.shape}/{ta.type_name} -> "
                    f"{tb.name} {tb.shape}/{tb.type_name}"
                )

    if problems:  # per-tensor digests are meaningless once the shape lists diverge
        return problems, 0

    differing: list[str] = []
    for ta, tb in zip(a.tensors, b.tensors):
        if ta.n_bytes != tb.n_bytes:
            differing.append(ta.name)
            continue
        if tensor_digest(src, a.data_offset, ta.offset, ta.n_bytes) != \
           tensor_digest(dst, b.data_offset, tb.offset, tb.n_bytes):
            differing.append(ta.name)
    if differing:
        problems.append(f"{len(differing)} tensor(s) differ in content: {differing[:6]}")
    return problems, len(a.tensors)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verify_rewrite", description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args(argv)

    try:
        problems, compared = compare(args.src, args.dst)
    except (GGUFError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"src           {os.path.basename(args.src)} "
          f"({os.path.getsize(args.src)} bytes)")
    print(f"dst           {os.path.basename(args.dst)} "
          f"({os.path.getsize(args.dst)} bytes)")
    print(f"tensors       {compared} compared by content digest")
    if problems:
        for line in problems:
            print(f"DIFFERS       {line}")
        print("verdict       NOT lossless")
        return 1
    print("verdict       lossless: metadata, descriptors and every tensor identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
