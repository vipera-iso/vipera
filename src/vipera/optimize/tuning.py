"""Remember what this machine actually ran a model at, and reuse it.

The planner derives ``gpu_layers`` from a VRAM budget that is a formula, not a
measurement: it assumes a fixed reserve for the CUDA context, a bytes-per-layer
figure from the quant, and a KV size from the dtype. On the reference 4 GB card
the estimate was one layer optimistic for one model and one pessimistic for
another, and a failed layer shows up as a full reload at the next rung.

The measurement itself is cheap once a model is loaded, and ``vipera bench``
already does it. This module is the memory for that: a profile per (model,
machine fingerprint) holding the split that produced the best measured decode
throughput, so the next start begins from what worked instead of from the
formula. Nothing here runs a model or guesses — an absent profile changes
nothing.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vipera.config import TUNING_FILE, quarantine_corrupt
from vipera.planner import Plan

SCHEMA_VERSION = 1

# Measurement-time noise on a card's free VRAM (another process finishing, the
# driver's own bookkeeping). Below this, a profile still applies.
_FREE_VRAM_SLACK_GB = 0.25


@dataclass(frozen=True)
class Tuning:
    """One measured profile: what loaded, and how fast it decoded.

    ``free_gb`` is the VRAM that was free when it loaded. It is not part of the
    fingerprint (it moves with whatever else is running), but a split measured
    with room to spare must not be replayed after another process took it.
    """

    gpu_layers: int
    ctx: int = 0
    gen_tps: float = 0.0
    measured_at: str = ""
    free_gb: float = 0.0
    # The tensor placement this profile measured, -1 for "not a tensor split",
    # exactly like Plan's own fields. A server plan's offload is `-ncmoe` or
    # `-ncffn`, and storing it is the point: the planner's fit is a formula while
    # the split that loaded is a measurement, and replaying it is what stops
    # every start from paying the probe ladder again (up to ten spawns, ~1.5 s
    # each, on a dense 7B). Absent keys read as -1, so files written before this
    # field existed still load.
    n_cpu_ffn: int = -1
    moe_cpu_layers: int = -1
    # The micro-batch that split loaded at. A probe that had to drop to 256 to
    # create a context has to be replayed at 256, or the replay fails and the
    # ladder runs from the top again.
    n_ubatch: int = 0


def fingerprint(hw: Any) -> str:
    """Identity of the machine a profile belongs to.

    GPU name plus total VRAM separates the cards Vipera cares about; cores and
    total RAM separate the hosts behind the same card. Free VRAM is deliberately
    absent — it changes with whatever else is running, and a profile that
    stopped matching after another process took 200 MB would be worse than none.
    """
    gpu = getattr(hw, "gpu_name", "None") or "None"
    return (
        f"{gpu}|{getattr(hw, 'vram_total_gb', 0.0):.0f}GB|"
        f"{getattr(hw, 'cpu_cores', 0)}c|{getattr(hw, 'ram_total_gb', 0.0):.0f}GB"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class TuningStore:
    """`tuning.json` in the Vipera home: model -> machine -> measured profile."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else TUNING_FILE
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            quarantine_corrupt(self.path, exc)
            return {}
        if not isinstance(data, dict):
            quarantine_corrupt(self.path, ValueError("not a JSON object"))
            return {}
        profiles = data.get("profiles")
        return profiles if isinstance(profiles, dict) else {}

    def get(self, alias: str, hw: Any) -> Tuning | None:
        entry = self._read().get(alias, {}).get(fingerprint(hw))
        if not isinstance(entry, dict):
            return None
        try:
            return Tuning(
                gpu_layers=int(entry["gpu_layers"]),
                ctx=int(entry.get("ctx") or 0),
                gen_tps=float(entry.get("gen_tps") or 0.0),
                measured_at=str(entry.get("measured_at") or ""),
                free_gb=float(entry.get("free_gb") or 0.0),
                n_cpu_ffn=int(entry.get("n_cpu_ffn", -1)),
                moe_cpu_layers=int(entry.get("moe_cpu_layers", -1)),
                n_ubatch=int(entry.get("n_ubatch") or 0),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def record(
        self,
        alias: str,
        hw: Any,
        *,
        gpu_layers: int,
        ctx: int = 0,
        gen_tps: float = 0.0,
        n_cpu_ffn: int = -1,
        moe_cpu_layers: int = -1,
        n_ubatch: int = 0,
    ) -> bool:
        """Store a measurement; keep the fastest one seen for this machine.

        Returns True when the profile changed, so a caller can say so instead of
        claiming a save that did nothing.
        """
        if gpu_layers < 0 or gen_tps <= 0:
            return False
        with self._lock:
            profiles = self._read()
            model = profiles.get(alias)
            if not isinstance(model, dict):
                model = {}
            key = fingerprint(hw)
            previous = model.get(key)
            if isinstance(previous, dict):
                try:
                    if float(previous.get("gen_tps") or 0.0) >= gen_tps:
                        return False
                except (TypeError, ValueError):
                    pass
            model[key] = {
                "gpu_layers": int(gpu_layers),
                "ctx": int(ctx),
                "gen_tps": round(float(gen_tps), 3),
                "measured_at": _now(),
                "free_gb": round(float(getattr(hw, "vram_free_gb", 0.0) or 0.0), 2),
                "n_cpu_ffn": int(n_cpu_ffn),
                "moe_cpu_layers": int(moe_cpu_layers),
                "n_ubatch": int(n_ubatch),
            }
            profiles[alias] = model
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"_schema": SCHEMA_VERSION, "profiles": profiles}, indent=2))
            tmp.replace(self.path)
            return True

    def entries(self) -> dict[str, dict[str, Any]]:
        """Everything stored, for `vipera system config`-style inspection."""
        return self._read()


def apply_tuning(
    p: Plan,
    alias: str,
    hw: Any,
    store: TuningStore | None = None,
) -> Plan:
    """Return ``p`` with the measured placement, or ``p`` unchanged.

    Two kinds of profile are replayed. A classic plan learns its ``gpu_layers``.
    A server plan learns the tensor split it measured (``-ncffn``/``-ncmoe`` plus
    the micro-batch it loaded at): the planner's fit is a formula, the probe
    ladder that turns it into a measurement costs a spawn per rung on every
    start, and a stored split is confirmed by the first load attempt instead - so
    a stale profile still walks down the safe rungs. That holds when the profile
    agrees with the fit too: the split is measured either way, even when
    replaying it changes nothing about the plan.
    """
    profile = (store or TuningStore()).get(alias, hw)
    if profile is None:
        return p
    # Free VRAM at load time is what the split has to fit into. If the card has
    # less room now than when it was measured, the formula's smaller split is
    # the honest answer: the measured one would only fail and walk back down.
    free_now = getattr(hw, "vram_free_gb", None)
    if (
        profile.free_gb > 0
        and free_now is not None
        and float(free_now) + _FREE_VRAM_SLACK_GB < profile.free_gb
    ):
        return p
    # The KV cache scales with the window, and the profile was measured at one
    # specific context. Replaying an 8K split into a 64K plan over-allocates by
    # the difference, which is exactly a failed context on a 4 GB card; the
    # formula's smaller split is the honest answer there.
    if profile.ctx > 0 and p.context > profile.ctx:
        return p
    if p.requires_server:
        return _apply_tensor_split(p, profile)
    layers = profile.gpu_layers
    if p.n_layers > 0:
        layers = min(layers, p.n_layers)
    layers = max(0, layers)
    if layers == p.gpu_layers:
        return p
    rate = f", {profile.gen_tps:.2f} t/s" if profile.gen_tps else ""
    return replace(p, gpu_layers=layers, note=f"{p.note} (measured: {layers} layers{rate})")


def _apply_tensor_split(p: Plan, profile: Tuning) -> Plan:
    """Replay a measured -ncffn / -ncmoe split, or return ``p`` unchanged.

    Only when the planner itself chose a tensor placement: the profile says
    *where* that split landed, not that a tensor split applies at all - a 2K
    window or a model without per-tensor sizes made the planner pick the classic
    split deliberately, and replaying over that decision would undo its reason.
    """
    if p.n_layers <= 0:
        return p
    ubatch = profile.n_ubatch or p.ubatch
    if profile.n_cpu_ffn >= 0 and p.n_cpu_ffn >= 0:
        n_cpu_ffn = max(0, min(profile.n_cpu_ffn, p.n_layers))
        if n_cpu_ffn == p.n_cpu_ffn and ubatch == p.n_ubatch:
            return replace(p, measured=True)
        gpu_ffn = p.n_layers - n_cpu_ffn
        return replace(
            p,
            n_cpu_ffn=n_cpu_ffn,
            n_ubatch=ubatch,
            measured=True,
            note=f"{p.note} (measured: {gpu_ffn}/{p.n_layers} dense FFN layers on GPU, ub {ubatch})",
        )
    if profile.moe_cpu_layers >= 0 and p.moe_cpu_layers >= 0:
        moe_cpu_layers = max(0, min(profile.moe_cpu_layers, p.n_layers))
        if moe_cpu_layers == p.moe_cpu_layers and ubatch == p.n_ubatch:
            return replace(p, measured=True)
        gpu_experts = p.n_layers - moe_cpu_layers
        return replace(
            p,
            moe_cpu_layers=moe_cpu_layers,
            n_ubatch=ubatch,
            measured=True,
            note=f"{p.note} (measured: MoE, {gpu_experts}/{p.n_layers} expert layers on GPU)",
        )
    return p
