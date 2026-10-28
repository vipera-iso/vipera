"""Measure how concentrated a model's FFN activations are (stage D, docs/REWRITE.md).

Stage D is the only lever that raises the ALU ceiling: PowerInfer-style activation
sparsity skips FFN work per token. This module measures the one activation-level
quantity llama.cpp can produce without new instrumentation — `llama-imatrix`'s
per-input-channel sum of squared activations — and turns it into the numbers a
decision needs:

* how much of the FFN's activation energy sits in the top decile of channels,
* how many channels are effectively dead (below a share of the median),
* the participation ratio, i.e. how many channels carry the energy.

What this is **not**: per-token sparsity. An imatrix aggregates over every token in
the corpus, so it cannot show that a *given token* reads few channels — and that is
the property a sparse kernel needs. The report says so rather than letting a static
number pass for a dynamic one.

    python3 -m bench.sparsity --imatrix imatrix.gguf \\
        --tensor blk.0.ffn_down_exps.weight.in_sum2

Stdlib only, like the rest of vipera (pyproject: `dependencies = []`).

Exit codes: 0 measured, 1 the tensor is missing or unreadable, 2 bad usage.
"""

from __future__ import annotations

import argparse
import array
import os
import sys

from vipera.gguf import GGUFError, read_gguf

F32 = 0  # GGML_TYPE_F32


def read_tensor_f32(path: str, name: str) -> tuple[tuple[int, ...], list[float]]:
    """Read one F32 tensor out of a GGUF's data section."""
    g = read_gguf(path)
    t = g.tensor(name)
    if t is None:
        raise GGUFError(f"{os.path.basename(path)}: no tensor named {name!r}")
    if t.ggml_type != F32:
        raise GGUFError(f"{name!r} is {t.type_name}, not F32; cannot read it as energies")
    with open(path, "rb") as f:
        f.seek(g.data_offset + t.offset)
        raw = f.read(t.n_bytes)
    if len(raw) != t.n_bytes:
        raise GGUFError(f"{name!r}: wanted {t.n_bytes} bytes, read {len(raw)}")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":  # GGUF is little-endian
        values.byteswap()
    return t.shape, list(values)


def _columns(values: list[float], shape: tuple[int, ...]) -> tuple[list[list[float]], str]:
    """Split a flat per-channel energy tensor into one vector per expert.

    imatrix writes a MoE expert tensor as `(n_in, n_experts)`, and GGUF stores `ne0`
    contiguously, so **expert `e` is one contiguous block** of `ne0` channel values:
    `values[e * n_in : (e + 1) * n_in]`. This is worth stating because the wrong
    reading - striding by `n_experts` instead - is silently plausible.

    Measured on `DeepSeek-V2-Lite` (`blk.13.ffn_down_exps.weight.in_sum2`), the block
    reading gives a between-expert CV of 1.22 against 0.08 for the strided one, and
    the same 5-15x gap holds on every expert tensor checked: routing skew makes the
    real experts differ from each other, while a strided mixture averages that away.
    """
    if len(shape) == 2 and shape[1] > 1 and len(values) == shape[0] * shape[1]:
        n_in, n_experts = int(shape[0]), int(shape[1])
        return ([values[e * n_in:(e + 1) * n_in] for e in range(n_experts)],
                "per expert (contiguous blocks)")
    return [list(values)], "all experts summed"


def channel_stats(energy: list[float]) -> dict:
    """Summarize one activation vector of per-channel energies.

    `energy` is one expert's per-input-channel sums of squared activations,
    accumulated over the corpus.
    """
    if not energy:
        raise ValueError("empty activation vector")
    total = sum(energy)
    if total <= 0:
        raise ValueError("every channel has zero energy; nothing to measure")
    positive = [e if e > 0 else 0.0 for e in energy]
    descending = sorted((e / total for e in positive), reverse=True)
    n = len(descending)
    top_decile = max(1, int(round(0.10 * n)))
    ordered = sorted(positive)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    dead = sum(1 for e in positive if e < 0.01 * median) if median > 0 else 0
    # participation ratio: (sum e)^2 / sum e^2 — how many channels "carry" the energy
    participation = total * total / sum(e * e for e in positive)
    return {
        "channels": n,
        "top_decile_share": sum(descending[:top_decile]),
        "dead_channels": dead,
        "dead_fraction": dead / n,
        "participation_ratio": participation,
        "participation_fraction": participation / n,
        "nonzero_channels": sum(1 for e in positive if e > 0),
    }


def format_report(stats: dict, *, tensor: str, corpus_note: str) -> str:
    lines = [
        f"tensor        {tensor}",
        f"slice         {corpus_note}",
        f"channels      {stats['channels']} ({stats['nonzero_channels']} with any energy)",
        f"top decile    {100 * stats['top_decile_share']:.1f}% of the activation energy",
        f"dead channels {stats['dead_channels']} "
        f"({100 * stats['dead_fraction']:.1f}% below 1% of the median channel)",
        f"participation {stats['participation_ratio']:.1f} channels "
        f"({100 * stats['participation_fraction']:.1f}% of them carry the energy)",
        "",
        "reading       a *static* aggregate over the whole corpus. Per-token sparsity -",
        "              what a sparse FFN kernel skips - cannot be read from an imatrix,",
        "              and is not claimed here.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sparsity", description=__doc__)
    p.add_argument("--imatrix", required=True, help="imatrix GGUF written by llama-imatrix")
    p.add_argument("--tensor", required=True,
                   help="e.g. blk.0.ffn_down_exps.weight.in_sum2")
    p.add_argument("--expert", type=int, default=0,
                   help="which expert column to summarize (MoE tensors only)")
    p.add_argument("--all-experts", action="store_true",
                   help="sum every expert column instead of picking one")
    p.add_argument("--json", dest="json_path")
    args = p.parse_args(argv)

    try:
        shape, values = read_tensor_f32(args.imatrix, args.tensor)
    except (GGUFError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    columns, kind = _columns(values, shape)
    if args.all_experts or len(columns) == 1:
        energy = [sum(col) for col in zip(*columns)] if len(columns) > 1 else columns[0]
        note = f"all {len(columns)} experts summed (shape {shape})"
    else:
        if not 0 <= args.expert < len(columns):
            print(f"error: expert {args.expert} is outside 0..{len(columns) - 1}", file=sys.stderr)
            return 2
        energy = columns[args.expert]
        note = f"expert {args.expert} of {len(columns)} ({kind}, shape {shape})"

    try:
        stats = channel_stats(energy)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(format_report(stats, tensor=args.tensor, corpus_note=note))

    if args.json_path:
        import json

        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump({"tensor": args.tensor, "shape": list(shape), "slice": note, **stats},
                      f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
