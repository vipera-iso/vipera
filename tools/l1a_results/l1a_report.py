#!/usr/bin/env python3
"""L1a report: the 5 agreed numbers, computed from the per-run JSONs.

Fails loudly if any run JSON is missing/unparsable or if run params drifted
from the fixed set (ctx=512, chunks>=24, ngl=0) - numbers with drifted params
must not be compared.
"""

import json
import sys
from pathlib import Path

J = Path("/tmp/opencode")
BASELINE_RUN1 = 6.9289  # L5a baseline v1 (200-row sample), mem_mvglxymn
GATE_SESSION = 0.3  # #1 orig_run2 drift vs run1
GATE_IMATRIX = 0.25  # #3 l1a - control needs > ~1x noise (1 sigma ~0.22)
GATE_Q3 = 0.4  # #5 Q3 sanity gate

NEEDED = {
    "hermes_orig2": "NousResearch_Hermes-4-14B-Q4_K_M.gguf",
    "hermes_ctl_q4": "ctl-hermes-q4km.gguf",
    "hermes_l1a_q4": "l1a-hermes-q4km.gguf",
    "hermes_ctl_q3": "ctl-hermes-q3km.gguf",
    "hermes_l1a_q3": "l1a-hermes-q3km.gguf",
    "qwen_orig2": "qwen15b.gguf",
    "qwen_ctl_q4": "ctl-qwen15b-q4km.gguf",
    "qwen_l1a_q4": "l1a-qwen15b-q4km.gguf",
}


def load(name: str, expect_model: str) -> dict:
    path = J / f"l1a_ppl_{name}.json"
    if not path.is_file():
        sys.exit(f"THIEU JSON: {path.name}")
    rows = json.loads(path.read_text())
    if len(rows) != 1 or rows[0]["ppl"] is None:
        sys.exit(f"RUN FAILED/khong co PPL: {name} -> {rows}")
    r = rows[0]
    if r["model"] != expect_model:
        sys.exit(f"SAI MODEL trong {name}: {r['model']} != {expect_model}")
    if r["ctx"] != 512 or r["ngl"] != 0 or (r["chunks"] or 0) < 24:
        sys.exit(f"PARAMS DROI trong {name}: ctx={r['ctx']} ngl={r['ngl']} chunks={r['chunks']}")
    return r


R = {name: load(name, model) for name, model in NEEDED.items()}


def line(label: str, a: dict, b: dict, gate: float | None = None, invert: bool = False) -> float:
    """invert=True -> gate is an UPPER bound (|d| must stay BELOW gate = PASS)."""
    d = a["ppl"] - b["ppl"]
    verdict = ""
    if gate is not None:
        ok = abs(d) < gate if invert else abs(d) >= gate
        verdict = "  PASS" if ok else f"  FAIL (nguong {gate})"
    print(f"{label:<42}{a['ppl']:>9.4f} - {b['ppl']:>9.4f} = {d:>+8.4f}{verdict}")
    return d


print("run thong so (phai giong nhau): ctx / ngl / chunks / wall_s")
for k in NEEDED:
    r = R[k]
    print(
        f"  {k:<16} ctx={r['ctx']} ngl={r['ngl']} chunks={r['chunks']} "
        f"wall={r['wall_s']}s sha256={r['sha256'][:12]}"
    )

print("\n5 CON SO:")
n1 = line(
    "#1 session: orig_run2 - run1(6.9289)",
    R["hermes_orig2"],
    {"ppl": BASELINE_RUN1},
    GATE_SESSION,
    invert=True,
)
print()
line("#2 requant loss: control - orig_run2", R["hermes_ctl_q4"], R["hermes_orig2"])
n3 = line(
    "#3 HIEU UNG IMATRIX (chinh): l1a - control",
    R["hermes_l1a_q4"],
    R["hermes_ctl_q4"],
    GATE_IMATRIX,
)
line("#4 net: l1a - orig_run2", R["hermes_l1a_q4"], R["hermes_orig2"])
n5 = line("#5 Q3 GATE: q3_l1a - q3_control", R["hermes_l1a_q3"], R["hermes_ctl_q3"], GATE_Q3)

print("\nHANG PHU (qwen15b, khong cham gate):")
line("qwen requant loss: control - orig_run2", R["qwen_ctl_q4"], R["qwen_orig2"])
line("qwen hieu ung imatrix: l1a - control", R["qwen_l1a_q4"], R["qwen_ctl_q4"])

print("\nQUYET DINH:")
if abs(n1) > GATE_SESSION:
    print(f"  DUNG: session noise {n1:+.4f} > {GATE_SESSION} - may khong on dinh, #2-#5 vo nghia")
elif abs(n5) < GATE_Q3:
    print(f"  DUNG: Q3 gate {n5:+.4f} < {GATE_Q3} - khong ket luan Q4, dieu truoc khi dung L1b")
    print("  (nho loai tru hypothesis: imatrix build tu Q4_K_M khong phai F16)")
else:
    note = (
        "null result - pipeline OK, Q4_K_M da toi uu"
        if abs(n3) < GATE_IMATRIX
        else "imatrix co hieu ung ro rang o Q4_K_M"
    )
    print(f"  Tiep tuc: Q3 gate pass ({n5:+.4f}), #3 = {n3:+.4f} -> {note}")
