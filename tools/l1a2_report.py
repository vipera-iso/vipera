#!/usr/bin/env python3
"""L1a v3 report: imatrix effect when the source is actually re-quantized.

Pairs are only comparable inside one group (same source file, same eval,
same llama-perplexity). Prints the absolute and relative PPL delta of every
imatrix arm against its no-imatrix control, plus the run parameters and the
perplexity stderr as the resolution limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

J = Path(__file__).resolve().parent / "l1a2_results"

GROUPS = {
    "Q4_K_M (nguon Q8_0)": [
        ("ctl", "q4_ctl.json"),
        ("imatrix BF16 (bartowski)", "q4_imbf.json"),
    ],
    "Q3_K_M (nguon Q8_0)": [
        ("ctl", "q3_ctl.json"),
        ("imatrix BF16 (bartowski)", "q3_imbf.json"),
    ],
}


def load(name: str) -> dict:
    path = J / name
    if not path.is_file():
        sys.exit(f"THIEU JSON: {path}")
    rows = json.loads(path.read_text())
    if len(rows) != 1 or rows[0]["ppl"] is None:
        sys.exit(f"RUN FAILED: {name} -> {rows}")
    r = rows[0]
    if r["ctx"] != 512 or r["ngl"] != 0 or (r["chunks"] or 0) < 24:
        sys.exit(f"PARAMS DROI: {name}: ctx={r['ctx']} ngl={r['ngl']} chunks={r['chunks']}")
    return r


print("run params (phai giong nhau moi arm):")
for files in GROUPS.values():
    for _label, f in files:
        r = load(f)
        print(
            f"  {f:<16}{r['model'][:34]:<36} ctx={r['ctx']} ngl={r['ngl']} "
            f"chunks={r['chunks']} tok={r['scored_tok']} ppl={r['ppl']:.4f}"
        )

for group, arms in GROUPS.items():
    base = load(arms[0][1])
    err = base["ppl_stderr"] or 0.0
    print(f"\n{group}  (ctl PPL {base['ppl']:.4f} ± {err:.4f}, {base['scored_tok']} token)")
    for label, f in arms[1:]:
        r = load(f)
        d = r["ppl"] - base["ppl"]
        print(
            f"  {label:<26}{r['ppl']:>9.4f}  Δ = {d:>+8.4f} "
            f"({d / base['ppl'] * 100:>+6.2f}%)  {'< 1 stderr' if abs(d) < err else '>= 1 stderr'}"
        )

print(
    "\nDoc so:\n"
    "  - Δ < 0 la imatrix lam PPL giam (tot hon).\n"
    "  - stderr cua PPL la do eval set (24 chunk ~ 12k token), khong phai nhieu\n"
    "    run-to-run (run lap lai giong tung so le) => |Δ| < stderr la khong ket luan\n"
    "    duoc la co loi tren van ban khac.\n"
)
