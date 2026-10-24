#!/usr/bin/env python3
"""Audit a llama-quantize run: did it actually quantize anything?

L1a's Q4 arm compared `Q4_K_M -> Q4_K_M` and got a PPL delta of exactly
+0.0000. That is not a null result: llama-quantize copies a tensor verbatim
when the source and target types are equal (`quantize = cur_type != new_type`
in llama-quant.cpp), so the importance matrix is never even looked up. A
quantize run must be proven to have changed the weights before its PPL number
means anything.

Usage:
    python tools/gguf_quant_audit.py --src A.gguf --out B.gguf
        compare per-tensor types and data hashes: is B a real re-quantization?

    python tools/gguf_quant_audit.py --a ctl.gguf --b l1a.gguf --same-source
        compare two outputs of the same source: did the imatrix change values?

Exit codes: 0 = audit ran, 3 = --src/--out run was a no-op (same types or
identical data for every tensor).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    from gguf import GGUFReader
except ImportError:  # pragma: no cover - dependency is optional in the repo
    sys.exit("gguf is not installed (pip install 'vipera-ai[gguf]')")

CHUNK = 1 << 20


def load(path: str) -> tuple[GGUFReader, Path]:
    p = Path(path).expanduser()
    if not p.is_file():
        sys.exit(f"not found: {p}")
    return GGUFReader(str(p)), p


def tensor_types(reader: GGUFReader) -> dict[str, str]:
    return {t.name: t.tensor_type.name for t in reader.tensors}


def data_hashes(reader: GGUFReader, path: Path) -> dict[str, str]:
    """sha256 of each tensor's raw bytes, read straight from the file.

    tensor.data_offset is absolute, so the reader does not need to map the
    whole model into RAM: only one slice per tensor is read at a time.
    """
    out: dict[str, str] = {}
    with open(path, "rb") as f:
        for t in reader.tensors:
            digest = hashlib.sha256()
            f.seek(int(t.data_offset))
            remaining = int(t.n_bytes)
            while remaining:
                block = f.read(min(CHUNK, remaining))
                if not block:
                    sys.exit(f"short read on {t.name} in {path}")
                digest.update(block)
                remaining -= len(block)
            out[t.name] = digest.hexdigest()
    return out


def report_types(a: dict[str, str], b: dict[str, str], label: str) -> int:
    changed = [n for n in a if b.get(n) != a[n]]
    print(f"{label}: {len(changed)}/{len(a)} tensors changed type")
    for name in changed[:10]:
        print(f"    {name:<40} {a[name]:>10} -> {b.get(name, 'MISSING'):>10}")
    if len(changed) > 10:
        print(f"    ... and {len(changed) - 10} more")
    return len(changed)


def report_data(a: dict[str, str], b: dict[str, str], label: str) -> int:
    if set(a) != set(b):
        sys.exit(f"{label}: tensor sets differ, not comparable")
    changed = [n for n in a if a[n] != b[n]]
    print(f"{label}: {len(changed)}/{len(a)} tensors changed data")
    return len(changed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="quantize input")
    ap.add_argument("--out", help="quantize output")
    ap.add_argument("--a", help="first output of a shared source")
    ap.add_argument("--b", help="second output of a shared source")
    ap.add_argument("--skip-data", action="store_true", help="types only (fast)")
    args = ap.parse_args()

    if args.src and args.out:
        src, src_path = load(args.src)
        out, out_path = load(args.out)
        changed = report_types(tensor_types(src), tensor_types(out), f"{args.out} vs source")
        if args.skip_data:
            return 3 if changed == 0 else 0
        data_changed = report_data(data_hashes(src, src_path), data_hashes(out, out_path), "data")
        if changed == 0 or data_changed == 0:
            print(
                "\nNO-OP: this run never reached the quantizer, so an importance\n"
                "matrix could not have influenced it. Quantize from a higher-\n"
                "precision source (Q8_0/BF16), not from the target type."
            )
            return 3
        print("\nOK: real re-quantization.")
        return 0

    if args.a and args.b:
        a, a_path = load(args.a)
        b, b_path = load(args.b)
        report_types(tensor_types(a), tensor_types(b), "types a vs b")
        if args.skip_data:
            return 0
        report_data(data_hashes(a, a_path), data_hashes(b, b_path), "data a vs b")
        return 0

    ap.error("pass --src/--out or --a/--b")


if __name__ == "__main__":
    raise SystemExit(main())
