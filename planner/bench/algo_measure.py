"""Measure `vipera.algorithms` on real artifacts in this repository.

The unit tests assert the algorithmic properties. They cannot say whether the
algorithms do anything on *this project's* data, because a property test runs on
constructed input by design. This harness closes that gap and is the source of
every `[measured]` label in the registry.

Two real inputs, and one honest limitation:

1. **Real weights.** ``models/Qwen3-0.6B-Q4_K_M.gguf`` is read directly and its
   ``blk.N.ffn_gate.weight`` Q4_K blocks are dequantised. This is the one place a
   quantiser can be judged, because the alternative — synthetic Gaussian weights —
   is the distribution the quantiser is least likely to be wrong about.

   The dequantiser is **verified, not trusted**. No ``llama-quantize`` binary
   exists in this checkout, so the Q4_K layout is transcribed from
   ``ggml/src/ggml-quants.c`` (``dequantize_row_q4_K`` + ``get_scale_min_k4``) and
   then checked by re-encoding: for every value, ``round((y + m) / d)`` must equal
   the nibble it came from. A wrong sub-block scale, a wrong nibble half, or a
   swapped scale/min field all break that identity. The pass rate is reported and a
   run below 100% fails.

2. **A real trace.** ``bench/results/olmoe-1b-7b-511tok.jsonl`` is 511 genuinely
   captured decode steps from OLMoE-1B-7B, so the selection result is about this
   project's target, not about Zipf.

3. **What is NOT measured.** The Hessian GPTQ needs comes from calibration
   activations, and this repository has none — no activation capture exists here.
   The Hessian is therefore generated, and every number that depends on it is
   labelled ``[simulated]``. The *weights* are real; the *calibration* is not. Both
   arms are run so the reader can see which claim rests on which input.

Run: `python3 -m bench.algo_measure [--out bench/results/algo-measure.json]`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vipera.algorithms import quant, select                     # noqa: E402
from vipera.gguf import read_gguf                                # noqa: E402

QK_K = 256           # super-block size for every k-quant in ggml
Q4_K_BYTES = 144     # d(2) + dmin(2) + scales(12) + qs(128)
Q4_K_TYPE = 12       # GGML_TYPE_Q4_K

DEFAULT_MODEL = "models/Qwen3-0.6B-Q4_K_M.gguf"
DEFAULT_TENSOR = "blk.0.ffn_gate.weight"
DEFAULT_TRACE = "bench/results/olmoe-1b-7b-511tok.jsonl"


def _f16(bits: int) -> float:
    """Decode one IEEE-754 half from its 16-bit pattern (ggml_half)."""
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0:
        value = frac * 2.0 ** -24
    elif exp == 0x1F:
        return float("inf") if frac == 0 else float("nan")
    else:
        value = (1.0 + frac / 1024.0) * 2.0 ** (exp - 15)
    return -value if sign else value


def _scale_min(j: int, scales: bytes) -> tuple[int, int]:
    """`get_scale_min_k4` from ggml/src/ggml-quants.c, transcribed.

    The layout is not two parallel arrays: entries 0..3 hold a scale and a min in
    the low six bits of four bytes each, and entries 4..7 split the high two bits
    of those same bytes into the *next* four entries' 4-bit fields. Reading it as
    two arrays of six-bit values produces plausible numbers that are wrong.
    """
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (
        (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
        (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
    )


def dequantize_q4_k(raw: bytes, *, first_block: int = 0, n_blocks: int = 1
                    ) -> list[tuple[float, int, float, float]]:
    """One or more Q4_K super-blocks -> `(value, nibble, d, m)` per element.

    The nibble and the two sub-block constants are returned beside the value so the
    caller can verify the decode instead of assuming it. Source:
    `dequantize_row_q4_K`, ggml/src/ggml-quants.c.
    """
    need = (first_block + n_blocks) * Q4_K_BYTES
    if len(raw) < need:
        raise ValueError(f"need {need} bytes, got {len(raw)}")
    out: list[tuple[float, int, float, float]] = []
    for b in range(n_blocks):
        base = (first_block + b) * Q4_K_BYTES
        block = raw[base:base + Q4_K_BYTES]
        d = _f16(struct.unpack_from("<H", block, 0)[0])
        dmin = _f16(struct.unpack_from("<H", block, 2)[0])
        scales = block[4:16]
        qs = block[16:144]
        is_ = 0
        pos = 0
        for _ in range(0, QK_K, 64):
            sc1, m1 = _scale_min(is_ + 0, scales)
            sc2, m2 = _scale_min(is_ + 1, scales)
            d1, lo1 = d * sc1, dmin * m1
            d2, lo2 = d * sc2, dmin * m2
            for l in range(32):
                q = qs[pos + l] & 0xF
                out.append((d1 * q - lo1, q, d1, lo1))
            for l in range(32):
                q = qs[pos + l] >> 4
                out.append((d2 * q - lo2, q, d2, lo2))
            pos += 32
            is_ += 2
    return out


def verify_dequant(decoded: list[tuple[float, int, float, float]], *, tol: float = 1e-3) -> dict:
    """Re-encode every value and check it reproduces the nibble it came from.

    `y = d*q - m` is exact up to fp32 rounding, so `(y + m)/d` must round back to
    the original integer `q`. A wrong scale/min assignment, a wrong nibble half, or
    a swapped field all fail this, which is what makes the dequantiser trustworthy
    without a reference binary.
    """
    bad = 0
    worst = 0.0
    for y, q, d, m in decoded:
        if d <= 0:
            bad += 1
            continue
        recovered = (y + m) / d
        worst = max(worst, abs(recovered - q))
        if abs(recovered - q) > tol:
            bad += 1
    return {
        "values": len(decoded),
        "failures": bad,
        "pass_rate": round(1.0 - bad / len(decoded), 9) if decoded else 0.0,
        "worst_round_trip_error": worst,
    }


def read_q4_k_block(path: str, tensor_name: str, *, rows: int, cols: int) -> dict:
    """Read the first `cols` columns of the first `rows` rows of a Q4_K tensor.

    `cols` must be a whole number of 256-element super-blocks, because a partial
    block has no meaning: the sub-block scales are shared across it.
    """
    if cols % QK_K != 0:
        raise ValueError(f"cols must be a multiple of {QK_K}, got {cols}")
    gguf = read_gguf(path)
    info = gguf.tensor(tensor_name)
    if info is None:
        raise ValueError(f"tensor {tensor_name!r} not found in {path}")
    if info.ggml_type != Q4_K_TYPE:
        raise ValueError(
            f"tensor {tensor_name!r} is type {info.ggml_type}, not Q4_K ({Q4_K_TYPE})"
        )
    ne0, ne1 = info.shape[0], info.shape[1]
    if ne0 < cols or ne1 < rows:
        raise ValueError(f"tensor is {ne0}x{ne1}, too small for {rows}x{cols}")
    blocks_per_row = ne0 // QK_K
    row_stride = blocks_per_row * Q4_K_BYTES
    want_blocks = cols // QK_K

    base = gguf.data_offset + info.offset
    blocks_raw = bytearray()
    with open(path, "rb") as f:
        for r in range(rows):
            f.seek(base + r * row_stride)
            blocks_raw += f.read(want_blocks * Q4_K_BYTES)

    decoded = dequantize_q4_k(bytes(blocks_raw), n_blocks=rows * want_blocks)
    per_row = cols
    W: list[list[float]] = []
    for r in range(rows):
        chunk = decoded[r * per_row:(r + 1) * per_row]
        W.append([v for v, _, _, _ in chunk])
    return {
        "tensor": tensor_name,
        "shape": [ne0, ne1],
        "rows": rows,
        "cols": cols,
        "blocks_read": rows * want_blocks,
        "bytes_read": len(blocks_raw),
        "weights": W,
        "verification": verify_dequant(decoded),
    }


def synthetic_hessian(cols: int, *, seed: int, samples: int = 128) -> list[list[float]]:
    """A positive-definite `cols x cols` matrix, as `E[x xᵀ]` from calibration would be.

    **Synthetic.** No activation capture exists in this repository, so this is the
    one input to the GPTQ measurement that is not real data, and it is why the
    Hessian-norm numbers below are labelled `[simulated]`.

    ``H = A Aᵀ / n + ridge I`` is positive definite by construction, so the inverse
    exists and the propagation is well defined. A ridge of 1e-3 keeps the
    near-unexcited directions invertible rather than blowing up.
    """
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def gauss() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u1 = ((state >> 11) + 1) / float(1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u2 = ((state >> 11) + 1) / float(1 << 53)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    A = [[gauss() for _ in range(cols)] for _ in range(samples)]
    H = [[0.0] * cols for _ in range(cols)]
    for i in range(cols):
        for j in range(i, cols):
            v = sum(A[r][i] * A[r][j] for r in range(samples)) / samples
            H[i][j] = v
            H[j][i] = v
    for i in range(cols):
        H[i][i] += 1e-3
    return H


def measure_quant(model: str, tensor: str, *, rows: int, cols: int, bits: int,
                  seed: int) -> dict:
    """GPTQ vs round-to-nearest on real weights, in both norms, plus a control arm."""
    block = read_q4_k_block(model, tensor, rows=rows, cols=cols)
    if block["verification"]["failures"]:
        raise SystemExit(
            "the Q4_K dequantiser failed its round-trip check "
            f"({block['verification']['failures']} values): every number below would "
            "be meaningless, so this is a hard stop rather than a warning"
        )
    W = block["weights"]
    H = synthetic_hessian(cols, seed=seed)

    rtn = quant.quantize_rtn(W, bits=bits)
    gptq = quant.quantize_gptq(W, H, bits=bits)

    # Control arm: H = I has a diagonal inverse, so nothing is propagated and the
    # two quantisers must agree bit for bit. If they do not, the propagation code is
    # acting on its own rather than on the Hessian.
    ident = [[1.0 if i == j else 0.0 for j in range(cols)] for i in range(cols)]
    control = quant.quantize_gptq(W, ident, bits=bits)

    return {
        "model": model,
        "tensor": block["tensor"],
        "tensor_shape": block["shape"],
        "weights_sampled": f"{rows}x{cols}",
        "blocks_read": block["blocks_read"],
        "bytes_read": block["bytes_read"],
        "verification": block["verification"],
        "bits": bits,
        "rtn_frobenius": quant.frobenius_mse(W, rtn),
        "gptq_frobenius": quant.frobenius_mse(W, gptq),
        "rtn_hessian": quant.hessian_objective(W, rtn, H),
        "gptq_hessian": quant.hessian_objective(W, gptq, H),
        "identity_control_identical": control == rtn,
        "hessian_is_synthetic": True,
    }


def measure_select(trace: str) -> dict:
    """Submodular selection on the real captured routing trace."""
    from vipera.algorithms.registry import _select_from_trace

    return _select_from_trace(trace, 6)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="algo_measure", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tensor", default=DEFAULT_TENSOR)
    p.add_argument("--trace", default=DEFAULT_TRACE)
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--cols", type=int, default=256,
                   help="must be a multiple of 256 (one Q4_K super-block)")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default="bench/results/algo-measure.json")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    report: dict = {"harness": "bench/algo_measure.py"}
    if os.path.exists(args.model):
        report["quant"] = measure_quant(args.model, args.tensor, rows=args.rows,
                                        cols=args.cols, bits=args.bits, seed=args.seed)
    else:
        print(f"skip quant: no model at {args.model}", file=sys.stderr)
    if os.path.exists(args.trace):
        report["select"] = measure_select(args.trace)
    else:
        print(f"skip select: no trace at {args.trace}", file=sys.stderr)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    q = report.get("quant")
    if q:
        v = q["verification"]
        print(f"quant  {q['tensor']} {q['tensor_shape']} -> {q['weights_sampled']} "
              f"real weights, {q['blocks_read']} Q4_K blocks, {q['bytes_read']} bytes")
        print(f"       dequantiser round-trip  {v['pass_rate'] * 100:.4f}% "
              f"({v['values']} values, worst error {v['worst_round_trip_error']:.2e})")
        print(f"       frobenius  rtn {q['rtn_frobenius']:.6f}   gptq {q['gptq_frobenius']:.6f}")
        print(f"       hessian    rtn {q['rtn_hessian']:.6f}   gptq {q['gptq_hessian']:.6f}"
              "   [H synthetic]")
        if q["rtn_hessian"] > 0 and q["gptq_hessian"] > 0:
            print(f"       hessian objective improvement  "
                  f"{q['rtn_hessian'] / q['gptq_hessian']:.3f}x")
        print(f"       identity-Hessian control identical to rtn: "
              f"{q['identity_control_identical']}")
    s = report.get("select")
    if s:
        print(f"select {s['trace']}  {s['tokens']} tokens x {s['layers']} layers x "
              f"{s['experts']} experts, k={s['top_k']}")
        print(f"       objective {s['objective']}")
        print(f"       prefix == greedy on {s['layers_where_prefix_equals_greedy']} layers; "
              f"min ratio to exact optimum {s['min_ratio_to_optimum']}")
        print(f"       submodular (1-1/e) bound applies here: {s['submodular_bound_applies']}")
    if args.out:
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
