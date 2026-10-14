"""Long context on a machine that cannot hold it: budget the KV, then slide.

Two facts decide this module.

1. KV grows *linearly* with context, so on a RAM-bound machine the context that
   fits is a hard budget, not a free setting. `arch.kv_bytes()` already models it
   (constant KDA state + linear MLA/Full attention); this module turns that into
   "the largest context this model + machine can hold", the number a run should
   not exceed without a plan.

2. llama.cpp can already generate *past* that budget. `--context-shift --keep N`
   keeps the first `N` tokens (the **attention sinks**) and slides a window over
   the rest, re-assigning positions inside the cache. That is the **StreamingLLM**
   mechanism (`2309.17453`). Its cost is exact: every token between the sinks and
   the window is dropped, so the model stops attending to the middle of a long
   document. llama.cpp ships it **disabled by default** and with `--keep 0`; a
   zero sink budget is the configuration StreamingLLM says breaks the model, so
   the useful part is choosing `N`.

What this module is *not*: H2O (`2306.14048`) evicts by accumulated attention
score, which llama.cpp has no code path for — that is a runtime patch, not a
planner, and is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .arch import Architecture

MIB = 1024 * 1024

# StreamingLLM keeps a handful of initial tokens as attention sinks. Four is the
# paper's own count and the safe floor: with zero sinks the model degrades
# sharply, which is why llama.cpp's `--keep 0` default is not usable as-is.
#
# Measured since (docs/REWRITE.md §B, wikitext-2, Qwen3-0.6B, 768-token window,
# 195 chunks): the sink *count* is worth more than the paper implies. Against the
# full contiguous window's PPL of 20.61, four sinks cost +4.05% and one sink
# +4.72% (both larger than the noise), while **32 sinks cost +0.64% - inside the
# noise**. So a caller who cares about fidelity should sweep `--sinks` rather than
# inherit the 4. The default stays 4 because this is one model and one corpus, and
# because four sinks is what the generation-stability result was established with;
# the measurement is a reason to make the choice explicit, not a new constant.
DEFAULT_SINKS = 4

# KV is usually placed on the same tier as the resident weights; leave the
# runtime a little room so the budget is not the whole machine.
DEFAULT_KV_HEADROOM_BYTES = 512 * MIB


@dataclass
class LongContextPolicy:
    """How to serve `requested_context` tokens with the KV this machine can hold."""

    requested_context: int
    window: int            # n_ctx actually allocated (the rolling window size)
    sinks: int             # --keep N: attention sinks pinned at the front
    context_shift: bool    # --context-shift: slide instead of stopping
    kv_bytes: int          # KV actually allocated for `window`
    cap: int               # largest context that fits without sliding
    reason: str
    warnings: list[str] = field(default_factory=list)

    @property
    def slides(self) -> bool:
        """True when the run will drop the middle of the context to keep going."""
        return self.context_shift

    def to_flags(self, *, kv_bits: int | None = None) -> list[str]:
        """The llama.cpp arguments this policy implies (`-ctk/-ctv` handled elsewhere)."""
        flags = ["-c", str(self.window)]
        if self.context_shift:
            flags += ["--keep", str(self.sinks), "--context-shift"]
        if kv_bits is not None:
            types = {16: "f16", 8: "q8_0", 4: "q4_0"}
            t = types.get(kv_bits)
            if t:
                flags += ["-ctk", t, "-ctv", t]
        return flags


def kv_bytes_per_token(arch: Architecture, *, kv_bits: int = 16) -> int:
    """Slope of the KV cache: bytes added per token (independent of context)."""
    bpp = kv_bits / 8
    return arch.kv_bytes(2, bytes_per_param=bpp) - arch.kv_bytes(1, bytes_per_param=bpp)


def resident_context_cap(
    *,
    arch: Architecture,
    model_bytes: int,
    ram_bytes: int,
    kv_bits: int = 16,
    headroom_bytes: int = DEFAULT_KV_HEADROOM_BYTES,
) -> int:
    """Largest context whose KV still fits beside `model_bytes` in RAM.

    Returns 0 when the model plus the constant KV state already exceeds the
    budget — i.e. no context beyond the minimum can be held.
    """
    bpp = kv_bits / 8
    fixed = arch.kv_bytes(0, bytes_per_param=bpp)
    slope = kv_bytes_per_token(arch, kv_bits=kv_bits)
    budget = ram_bytes - headroom_bytes - model_bytes
    if slope <= 0:
        return 0
    if budget <= fixed:
        return 0
    return (budget - fixed) // slope


def plan_long_context(
    *,
    arch: Architecture,
    model_bytes: int,
    ram_bytes: int,
    requested_context: int,
    kv_bits: int = 16,
    sinks: int = DEFAULT_SINKS,
    headroom_bytes: int = DEFAULT_KV_HEADROOM_BYTES,
    stream: bool = True,
) -> LongContextPolicy:
    """Decide the window, the sink budget and whether to slide.

    `stream` requests unbounded generation: even when the requested context fits,
    the policy arms context shift so a run is not stopped at the window edge.
    `sinks` is clamped to a quarter of the window so a small window still leaves
    room to attend to something.
    """
    if requested_context < 1:
        raise ValueError(f"requested_context must be >= 1, got {requested_context}")
    if sinks < 0:
        raise ValueError(f"sinks must be >= 0, got {sinks}")

    cap = resident_context_cap(
        arch=arch, model_bytes=model_bytes, ram_bytes=ram_bytes,
        kv_bits=kv_bits, headroom_bytes=headroom_bytes,
    )
    warnings: list[str] = []
    window = requested_context
    if cap and requested_context > cap:
        window = cap
        warnings.append(
            f"requested context {requested_context} exceeds the {cap} tokens the KV "
            f"budget holds; clamping the window to {cap}"
        )
    elif cap == 0:
        warnings.append(
            "no KV budget is left beside the model; the window is whatever llama.cpp "
            "allocates and the model will likely page"
        )

    sink_count = min(sinks, max(1, window // 4)) if stream else 0
    context_shift = bool(stream)

    bpp = kv_bits / 8
    kv_bytes = arch.kv_bytes(window, bytes_per_param=bpp)

    if context_shift:
        reason = (
            f"window {window} tokens with {sink_count} attention sinks and context "
            "shift (StreamingLLM 2309.17453): generation continues past the window "
            "by dropping the middle of the context"
        )
    else:
        reason = f"context {window} fits the KV budget with no sliding"

    return LongContextPolicy(
        requested_context=requested_context,
        window=window,
        sinks=sink_count,
        context_shift=context_shift,
        kv_bytes=kv_bytes,
        cap=cap,
        reason=reason,
        warnings=warnings,
    )


def describe(policy: LongContextPolicy, *, kv_bits: int = 16) -> str:
    """One line for a report."""
    flags = " ".join(policy.to_flags(kv_bits=kv_bits))
    kv_mib = policy.kv_bytes / MIB
    return (
        f"long context  {flags}  (KV {kv_mib:.0f} MiB, cap {policy.cap} tokens; "
        f"{policy.reason})"
    )
