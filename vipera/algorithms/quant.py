"""Weight quantization: round-to-nearest vs GPTQ-style error diffusion.

GPTQ (`2210.17323`) is the one thing in the literature pack that changes *how*
every weight is rounded rather than how many bits it gets. The paper's claim is
precise and is the claim this module tests:

    round-to-nearest minimizes the distance to each weight *individually*;
    GPTQ minimizes the reconstruction error in the **Hessian-weighted norm**
    ``E[‖(W − Ŵ)x‖²]`` — that is, the error the layer actually propagates.

Those are different objectives, so the honest comparison is **two** numbers, not
one. A quantizer that lowers the H-weighted norm can raise the plain Frobenius
error, and reporting only the metric the algorithm optimizes is exactly the
defect `docs/REVIEW.md` was written about. `compare()` returns both.

Two exact properties hold and are asserted in the tests rather than described:

* with ``H = I`` (or any diagonal ``H``) the off-diagonal inverse is zero, nothing
  is propagated, and GPTQ reduces **bit-for-bit** to round-to-nearest;
* the quantization *grid* is derived from the untouched weights in both, so a
  difference in error is a difference in diffusion and never in bit allocation.

Pure standard library, no numpy: the arrays are plain lists of lists. Gauss-Jordan
inversion is O(cols³), which is why this module is for a *reference* comparison on
layer-scale inputs, not for quantizing a 10 GB file.
"""

from __future__ import annotations

import math

# Symmetric quantization: with `bits` bits and a sign, the largest representable
# magnitude is (2^(bits-1) - 1) steps. Same convention as ggml's Q4_0.
def levels_for_bits(bits: int) -> int:
    """Number of steps from zero to the largest magnitude at `bits` bits."""
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    return 2 ** (bits - 1) - 1


def _round_half_away(x: float) -> float:
    """Round half away from zero, not Python's banker's rounding.

    `round(0.5) == 0` in Python. That silently breaks symmetry between +0.5 and
    -0.5, which shows up as a systematic bias once it is applied to millions of
    weights, so the quantizers must not use the builtin.
    """
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def _validate(W: list[list[float]]) -> tuple[int, int]:
    if not W:
        raise ValueError("weight matrix is empty")
    cols = len(W[0])
    if cols == 0:
        raise ValueError("weight matrix has zero columns")
    for r, row in enumerate(W):
        if len(row) != cols:
            raise ValueError(f"row {r} has {len(row)} entries, expected {cols}")
    return len(W), cols


def group_scales(row: list[float], *, levels: int, group: int) -> list[float]:
    """Per-group symmetric scale for one row, derived from the values as given.

    Clamped to a positive floor so an all-zero group cannot divide by zero and
    produce inf weights. The floor is tiny enough to be a no-op on real data.
    """
    scales: list[float] = []
    for g0 in range(0, len(row), group):
        chunk = row[g0:g0 + group]
        amax = max(abs(v) for v in chunk) if chunk else 0.0
        scales.append(amax / levels if amax > 0 else 1e-30)
    return scales


def quantize_rtn(
    W: list[list[float]], *, bits: int = 4, group: int | None = None
) -> list[list[float]]:
    """Round-to-nearest on the grid implied by `group`-sized abs-max scales."""
    rows, cols = _validate(W)
    levels = levels_for_bits(bits)
    g = cols if group is None else group
    if g < 1:
        raise ValueError(f"group must be >= 1, got {g}")
    out: list[list[float]] = []
    for row in W:
        scales = group_scales(row, levels=levels, group=g)
        q: list[float] = []
        for i, v in enumerate(row):
            s = scales[i // g]
            step = _round_half_away(v / s)
            step = max(-levels, min(levels, step))
            q.append(step * s)
        out.append(q)
    return out


def dense_inverse(A: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse with partial pivoting.

    Raises ValueError on a singular matrix instead of returning a matrix full of
    inf/nan: a singular Hessian means the calibration activations never excited
    some direction, and propagating a nan through the quantizer would look like a
    result.
    """
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError("dense_inverse needs a non-empty square matrix")
    M = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError(
                f"matrix is singular at column {col}: the Hessian gives no error "
                "weight there, so the layer cannot be scored"
            )
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        inv = 1.0 / pv
        M[col] = [v * inv for v in M[col]]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            Mr, Mc = M[r], M[col]
            M[r] = [a - factor * b for a, b in zip(Mr, Mc)]
    return [row[n:] for row in M]


def quantize_gptq(
    W: list[list[float]],
    H: list[list[float]],
    *,
    bits: int = 4,
    group: int | None = None,
    damp: float = 0.01,
) -> list[list[float]]:
    """GPTQ: quantize column by column, pushing each error into the columns left.

    `H` is the ``cols x cols`` Hessian of the layer input (``E[x xᵀ]`` from
    calibration activations). Its inverse supplies the direction each column's
    error is pushed along. `damp` adds a fraction of the mean diagonal to the
    diagonal before inverting — the paper's own guard, without which a nearly
    singular H makes the propagation explode.

    Error is propagated **within a group only**, because a group boundary is a
    change of scale: pushing a residual across it would move a value onto a grid
    its neighbour does not share.
    """
    rows, cols = _validate(W)
    if len(H) != cols or any(len(r) != cols for r in H):
        raise ValueError(f"H must be {cols}x{cols}, got {len(H)}x{len(H[0]) if H else 0}")
    if damp < 0:
        raise ValueError(f"damp must be >= 0, got {damp}")
    levels = levels_for_bits(bits)
    g = cols if group is None else group
    if g < 1:
        raise ValueError(f"group must be >= 1, got {g}")

    mean_diag = math.fsum(H[i][i] for i in range(cols)) / cols
    if mean_diag <= 0:
        raise ValueError("H has a non-positive mean diagonal; nothing to weight by")
    Hd = [[H[i][j] + (damp * mean_diag if i == j else 0.0) for j in range(cols)]
          for i in range(cols)]
    Hinv = dense_inverse(Hd)

    # The working copy carries the propagated error; `scales` is read from the
    # ORIGINAL weights so the grid is identical to round-to-nearest and the two
    # quantizers spend exactly the same bits.
    Wq = [list(row) for row in W]
    scales = [group_scales(row, levels=levels, group=g) for row in W]
    return _propagate(Wq, Hinv, scales, levels, g, rows, cols)


def _propagate(
    Wq: list[list[float]],
    Hinv: list[list[float]],
    scales: list[float],
    levels: int,
    g: int,
    rows: int,
    cols: int,
) -> list[list[float]]:
    """The column sweep: quantize column i, push its error into the rest of the group.

    `Wq` is the **working** matrix and is read before it is overwritten, because
    the residual that must be compensated is the one committed against the
    *corrected* column i — the value carried in from earlier columns — not against
    the original weight. Measuring it against the original (a first version of this
    function did) throws away exactly the information each earlier step pushed
    forward, and measurably makes GPTQ *worse* than round-to-nearest in both norms.

    One scale per row per group, taken from the untouched weights by the caller, so
    the grid is identical to round-to-nearest and the comparison is about diffusion
    alone.
    """
    for g0 in range(0, cols, g):
        g1 = min(g0 + g, cols)
        gi = g0 // g
        for i in range(g0, g1):
            inv_ii = Hinv[i][i]
            if inv_ii == 0.0:
                raise ValueError(f"Hessian inverse is singular at column {i}")
            error = [0.0] * rows
            for r in range(rows):
                s = scales[r][gi]
                v = Wq[r][i]
                step = _round_half_away(v / s)
                step = max(-levels, min(levels, step))
                q = step * s
                error[r] = v - q
                Wq[r][i] = q
            for j in range(i + 1, g1):
                coeff = Hinv[i][j] / inv_ii
                if coeff == 0.0:
                    continue
                for r in range(rows):
                    Wq[r][j] -= error[r] * coeff
    return Wq


def frobenius_mse(W: list[list[float]], Wq: list[list[float]]) -> float:
    """Plain mean squared error: what round-to-nearest minimizes."""
    rows, cols = _validate(W)
    total = 0.0
    for r in range(rows):
        for c in range(cols):
            d = W[r][c] - Wq[r][c]
            total += d * d
    return total / (rows * cols)


def hessian_objective(
    W: list[list[float]], Wq: list[list[float]], H: list[list[float]]
) -> float:
    """Mean Hessian-weighted error: what GPTQ claims to minimize.

    ``(1/rows) * sum_r (w_r - ŵ_r)ᵀ H (w_r - ŵ_r)``. This is the quantity that
    tracks the layer's output error, because a weight difference along a
    direction the input never excites costs nothing.
    """
    rows, cols = _validate(W)
    if len(H) != cols or any(len(r) != cols for r in H):
        raise ValueError(f"H must be {cols}x{cols}")
    total = 0.0
    for r in range(rows):
        w, q = W[r], Wq[r]
        # H is symmetric, so one triangle is enough: two dot products per row.
        acc = 0.0
        for i in range(cols):
            di = w[i] - q[i]
            if di == 0.0:
                continue
            row_sum = math.fsum(H[i][j] * (w[j] - q[j]) for j in range(cols))
            acc += di * row_sum
        total += acc
    return total / rows


def compare(
    W: list[list[float]], H: list[list[float]], *, bits: int = 4, group: int | None = None
) -> dict:
    """Both quantizers on one weight block, reported in both norms.

    The verdict field is deliberately about the *objective* GPTQ optimizes, and
    the Frobenius number is always printed beside it so a reader can see the cost
    rather than being told only the win.
    """
    rtn = quantize_rtn(W, bits=bits, group=group)
    gptq = quantize_gptq(W, H, bits=bits, group=group)
    return {
        "bits": bits,
        "rows": len(W),
        "cols": len(W[0]),
        "group": group if group is not None else len(W[0]),
        "rtn_frobenius": frobenius_mse(W, rtn),
        "gptq_frobenius": frobenius_mse(W, gptq),
        "rtn_hessian": hessian_objective(W, rtn, H),
        "gptq_hessian": hessian_objective(W, gptq, H),
    }
