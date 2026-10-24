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
    """Return ``p`` with the measured layer split, or ``p`` unchanged.

    Only the classic split is learned. A MoE plan's offload is ``-ncmoe``, which
    the server backend derives from ``n_layers``/``moe_cpu_layers``; the
    ``gpu_layers`` field is not what it uses, so a stored value would only look
    applied.
    """
    if p.requires_server:
        return p
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
    layers = profile.gpu_layers
    if p.n_layers > 0:
        layers = min(layers, p.n_layers)
    layers = max(0, layers)
    if layers == p.gpu_layers:
        return p
    rate = f", {profile.gen_tps:.2f} t/s" if profile.gen_tps else ""
    return replace(p, gpu_layers=layers, note=f"{p.note} (measured: {layers} layers{rate})")
