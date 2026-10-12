"""Plan where every bucket of weights should live: VRAM, RAM, or disk.

This is transform T2 from docs/OPTIMIZER.md. The plan is deterministic and
explainable — the runtime should never have to guess `-ngl` / `--n-cpu-moe`.

Policy (v0, size-driven):
  1. everything used on every token (attention, norms, router, shared experts,
     embeddings, output) goes to VRAM if it fits, otherwise to RAM;
  2. leftover VRAM becomes the expert cache;
  3. leftover RAM becomes resident experts;
  4. whatever is left spills to disk.

Step 2 is where routing traces (P2) will later rank *which* experts are hot. Without
traces the plan is uniform and says so in `placement.expert_order`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .analyze import Inventory

MIB = 1024 * 1024


@dataclass(frozen=True)
class Target:
    name: str
    vram_bytes: int
    ram_bytes: int
    disk: str = "unknown"
    # Leave this much headroom for KV cache, activations and the runtime itself.
    vram_headroom_bytes: int = 512 * MIB
    ram_headroom_bytes: int = 1536 * MIB
    # Memory tiers in GB/s (decimal). Defaults are the measured/derived ceilings
    # for the target laptop; they decide how long a byte costs, so they are part
    # of the machine description, not of the model.
    vram_bw_gbs: float = 192.0
    ram_bw_gbs: float = 40.0
    disk_bw_gbs: float = 2.0
    # Effective CPU throughput used by the scheduler's compute term (GFLOP/s).
    # Bandwidth-bound on this machine, but the compute floor still matters when
    # 18B parameters are active per token.
    cpu_gflops: float = 50.0

    @property
    def usable_vram(self) -> int:
        return max(0, self.vram_bytes - self.vram_headroom_bytes)

    @property
    def usable_ram(self) -> int:
        return max(0, self.ram_bytes - self.ram_headroom_bytes)

    @classmethod
    def from_toml(cls, path: str) -> "Target":
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            name=data.get("name", "unnamed"),
            vram_bytes=int(data["vram_mb"]) * MIB,
            ram_bytes=int(data["ram_mb"]) * MIB,
            disk=data.get("disk", "unknown"),
            vram_headroom_bytes=int(data.get("vram_headroom_mb", 512)) * MIB,
            ram_headroom_bytes=int(data.get("ram_headroom_mb", 1536)) * MIB,
            vram_bw_gbs=float(data.get("vram_bw_gbs", 192.0)),
            ram_bw_gbs=float(data.get("ram_bw_gbs", 40.0)),
            disk_bw_gbs=float(data.get("disk_bw_gbs", 2.0)),
            cpu_gflops=float(data.get("cpu_gflops", 50.0)),
        )


@dataclass
class Plan:
    manifest: dict
    warnings: list[str]
    # Computed directly rather than sniffed from warning prose, so rewording a
    # message can never silently flip the answer.
    fits: bool

    @property
    def runnable(self) -> bool:
        """True when the always-needed weights fit in VRAM+RAM at all."""
        return self.fits


def plan(inv: Inventory, target: Target, sharpness: float = 1.0) -> Plan:
    warnings: list[str] = []

    always = inv.always_resident_bytes
    vram = target.usable_vram
    ram = target.usable_ram

    fits = always <= vram + ram
    if not fits:
        warnings.append(
            f"always-resident weights ({always / MIB:.0f} MiB) do not fit in "
            f"VRAM+RAM ({vram / MIB:.0f} + {ram / MIB:.0f} MiB)"
        )
        resident_vram = min(always, vram)
        resident_ram = min(always - resident_vram, ram)
        leftover_vram = 0
        leftover_ram = 0
    else:
        resident_vram = min(always, vram)
        resident_ram = always - resident_vram
        leftover_vram = vram - resident_vram
        leftover_ram = ram - resident_ram

    expert = inv.expert_bytes
    cache_vram = min(expert, leftover_vram)
    remaining = expert - cache_vram
    expert_ram = min(remaining, leftover_ram)
    spill = remaining - expert_ram

    if spill > 0:
        warnings.append(
            f"{spill / MIB:.0f} MiB of experts spill to disk; on a DRAM-less NVMe "
            "expect roughly a 20x decode slowdown on cache misses"
        )

    cache_fraction = (cache_vram / expert) if expert else 0.0

    manifest = {
        "vipera_version": 1,
        "source": {
            "architecture": inv.architecture,
            "n_layers": inv.n_layers,
            "n_experts": inv.n_experts,
        },
        "target": {
            "name": target.name,
            "vram_mb": target.vram_bytes // MIB,
            "ram_mb": target.ram_bytes // MIB,
            "disk": target.disk,
            "usable_vram_mb": vram // MIB,
            "usable_ram_mb": ram // MIB,
        },
        "placement": {
            "resident_vram_bytes": resident_vram,
            "resident_ram_bytes": resident_ram,
            "expert_cache_vram_bytes": cache_vram,
            "expert_ram_bytes": expert_ram,
            "expert_spill_disk_bytes": spill,
            "expert_cache_fraction": round(cache_fraction, 4),
            # No traces yet: the hot set cannot be ranked, so the split is uniform.
            "expert_order": "uniform" if sharpness == 1.0 else "by_frequency",
            "expert_order_pending_trace": True,
        },
        "kv_cache": {"type_k": "q8_0", "type_v": "q8_0"},
    }

    return Plan(manifest=manifest, warnings=warnings, fits=fits)


def dumps(plan_result: Plan) -> str:
    return json.dumps(plan_result.manifest, indent=2) + "\n"
