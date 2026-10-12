"""Scoring for the LOCAL-LLM-16GB problem.

The problem defines

    S = (Q/100) * (T/10) * (16/M) * (65/P) * (32k/C)

where Q is quality as a percent of FP16, T tokens/s, M peak RAM in GB, P average
watts, and C the context that actually fits. A submission "solves" the problem
only if every hard constraint holds and S >= 1.

The formula is kept in one place so it is testable and so the CLI and any report
compute the same number.
"""

from __future__ import annotations

TARGET_TOK_S = 10.0
TARGET_RAM_GB = 16.0
TARGET_POWER_W = 65.0
TARGET_CONTEXT = 32768
TARGET_QUALITY_PCT = 100.0


def local_llm_16gb_score(
    *,
    quality_pct: float,
    tok_s: float,
    ram_gb: float,
    power_w: float,
    context: int,
) -> float:
    """The problem's composite score. Higher is better; >= 1 means solved."""
    if ram_gb <= 0 or power_w <= 0 or context <= 0:
        raise ValueError("ram_gb, power_w and context must be positive")
    return (
        (quality_pct / TARGET_QUALITY_PCT)
        * (tok_s / TARGET_TOK_S)
        * (TARGET_RAM_GB / ram_gb)
        * (TARGET_POWER_W / power_w)
        * (TARGET_CONTEXT / context)
    )
