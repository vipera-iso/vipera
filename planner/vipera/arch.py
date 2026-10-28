"""Per-layer architecture model for a GLM-5.3-Flash-style hybrid MoE.

Stage 2 of vipera: before any byte can be placed, the shape of the work must be
known *per layer*. A model like GLM-5.3-Flash is not one uniform stack — it
mixes two attention families, a sparse expert pool, a multi-stream residual
path and a multimodal encoder. Planning a streaming schedule for a flat
"45 layers x N bytes" blob would hide exactly the parts that decide feasibility.

This module is pure arithmetic over parameter counts. It never reads tensor data
and never loads a model, so it is cheap on the target machine and can be unit
tested against a toy config.

The default parameters in `arch/glm-5.3-flash.toml` are **derived**, not read
from the (unpublished) config: they were chosen so the totals reproduce the
publicly stated "321B total / 18B active" within a few percent. `calibrate()`
reports the residual so nobody mistakes the estimate for ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024 ** 3


@dataclass(frozen=True)
class MLA:
    """Multi-head Latent Attention (the 11 deep layers), DeepSeek-style.

    Weight-bearing projections: q, the kv compression pair (kv_a / kv_b), and o.
    The KV side is compressed to a low-rank latent, which is why MLA keeps a
    small cache even at 1M context.
    """

    n_heads: int = 48
    qk_nope: int = 128
    qk_rope: int = 64
    v_head: int = 128
    kv_lora: int = 512

    def params_per_layer(self, hidden: int) -> int:
        q = hidden * self.n_heads * (self.qk_nope + self.qk_rope)
        kv_a = hidden * (self.kv_lora + self.qk_rope)
        kv_b = self.kv_lora * self.n_heads * (self.qk_nope + self.v_head)
        o = self.n_heads * self.v_head * hidden
        return q + kv_a + kv_b + o

    def kv_bytes_per_token(self, bytes_per_param: float = 2.0) -> int:
        """Compressed KV written per token per MLA layer (latent + rope)."""
        return int((self.kv_lora + self.qk_rope) * bytes_per_param)


@dataclass(frozen=True)
class KDA:
    """Kimi Delta Attention (the 34 linear layers).

    Linear attention carries a constant-size recurrent state instead of a
    growing KV cache. Its weights are modelled as a multiple of hidden^2
    (q/k/v/o plus the decay gates and the short depthwise conv).
    """

    hidden_multiple: float = 7.0
    state_width: int = 128
    d_conv: int = 4

    def params_per_layer(self, hidden: int) -> int:
        return int(self.hidden_multiple * hidden * hidden)

    def state_bytes(self, n_heads: int = 48, bytes_per_param: float = 2.0) -> int:
        """Constant per-sequence state: heads x head_dim x head_dim, plus conv tail."""
        return int(n_heads * self.state_width * self.state_width * bytes_per_param)


@dataclass(frozen=True)
class Full:
    """Standard multi-head attention with grouped KV (GQA), as in Llama."""

    n_heads: int = 64
    n_kv_heads: int = 8
    head_dim: int = 128

    def params_per_layer(self, hidden: int) -> int:
        q = hidden * self.n_heads * self.head_dim
        k = hidden * self.n_kv_heads * self.head_dim
        v = hidden * self.n_kv_heads * self.head_dim
        o = self.n_heads * self.head_dim * hidden
        return q + k + v + o

    def kv_bytes_per_token(self, bytes_per_param: float = 2.0) -> int:
        return 2 * self.n_kv_heads * self.head_dim * int(bytes_per_param)


@dataclass(frozen=True)
class Architecture:
    name: str
    n_layers: int
    hidden: int
    vocab: int
    n_experts: int
    top_k: int
    moe_inter: int
    n_shared_experts: int = 1
    shared_inter: int = 1568
    kda_layers: int = 34
    mla_layers: int = 11
    full_layers: int = 0
    mhc_streams: int = 4
    mtp_layers: int = 1
    vision_params: int = 1_000_000_000
    # Leading blocks whose FFN is dense. They route nothing, so a routing trace
    # has `n_layers - leading_dense_layers` entries per token: the fork patch
    # records `ffn_moe_topk`, which exists only where there is a router. DeepSeek
    # and GLM both declare `first_k_dense_replace = 1`. `dense_inter` is that
    # block's FFN width and must be set whenever the count is non-zero.
    leading_dense_layers: int = 0
    dense_inter: int = 0
    # Published headline numbers, only used to report the calibration residual.
    stated_total: int = 0
    stated_active: int = 0
    mla: MLA = field(default_factory=MLA)
    kda: KDA = field(default_factory=KDA)
    full: Full = field(default_factory=Full)

    @property
    def dense(self) -> bool:
        """A dense model has no routed experts: one FFN per layer, always read."""
        return self.n_experts <= 1

    @property
    def moe_layers(self) -> int:
        """Blocks that route experts — the layer dimension a routing trace has."""
        return self.n_layers - self.leading_dense_layers

    def dense_ffn_params(self) -> int:
        """gate + up + down for the leading dense blocks; no router, no experts."""
        return self.leading_dense_layers * 3 * self.hidden * self.dense_inter

    # -- layer structure ---------------------------------------------------

    def layer_kinds(self) -> list[str]:
        """Per-layer attention kind, following the 3:1 super-hybrid cadence.

        `mla_layers` deep layers are placed every 4th position; the rest are
        KDA. Any leftover MLA layers are appended at the end so the counts are
        always honoured exactly.
        """
        if self.full_layers and not (self.kda_layers or self.mla_layers):
            if self.full_layers != self.n_layers:
                raise ValueError(
                    f"full_layers({self.full_layers}) != n_layers({self.n_layers})"
                )
            return ["full"] * self.n_layers
        if self.kda_layers + self.mla_layers != self.n_layers:
            raise ValueError(
                f"kda_layers({self.kda_layers}) + mla_layers({self.mla_layers}) "
                f"!= n_layers({self.n_layers})"
            )
        kinds: list[str] = []
        remaining = self.mla_layers
        for i in range(self.n_layers):
            if remaining > 0 and (i + 1) % 4 == 0:
                kinds.append("mla")
                remaining -= 1
            else:
                kinds.append("kda")
        # If the 3:1 stride did not place them all, fill from the end.
        i = self.n_layers - 1
        while remaining > 0 and i >= 0:
            if kinds[i] == "kda":
                kinds[i] = "mla"
                remaining -= 1
            i -= 1
        return kinds

    # -- per-component parameter counts ------------------------------------

    @property
    def expert_params_each(self) -> int:
        """gate + up + down for one routed expert."""
        return 3 * self.hidden * self.moe_inter

    def routed_expert_params(self) -> int:
        return self.n_experts * self.expert_params_each

    def shared_expert_params(self) -> int:
        return self.n_shared_experts * 3 * self.hidden * self.shared_inter

    def router_params(self) -> int:
        return self.hidden * self.n_experts

    def mhc_params(self) -> int:
        """Four-stream residual mixing: cheap per layer, but real."""
        return self.mhc_streams * self.hidden * 2

    def attn_params(self, kind: str) -> int:
        if kind == "mla":
            return self.mla.params_per_layer(self.hidden)
        if kind == "full":
            return self.full.params_per_layer(self.hidden)
        return self.kda.params_per_layer(self.hidden)

    # -- totals ------------------------------------------------------------

    def params_summary(self) -> dict[str, int]:
        kinds = self.layer_kinds()
        n_mla = kinds.count("mla")
        n_kda = kinds.count("kda")
        n_full = kinds.count("full")
        # Experts, shared experts and routers exist only on the MoE blocks; the
        # dense leading blocks carry one always-read FFN instead. With
        # `leading_dense_layers = 0` (every profile before this) the counts reduce
        # to the old `n_layers * ...` exactly, so no existing number moves.
        routed = self.moe_layers * self.routed_expert_params()
        dense = {"dense_ffn": self.dense_ffn_params()} if self.leading_dense_layers else {}
        return {
            "routed_experts": routed,
            "shared_experts": self.moe_layers * self.shared_expert_params(),
            **dense,
            "attention_kda": n_kda * self.kda.params_per_layer(self.hidden),
            "attention_mla": n_mla * self.mla.params_per_layer(self.hidden),
            "attention_full": n_full * self.full.params_per_layer(self.hidden),
            "router": self.moe_layers * self.router_params(),
            "residual_mhc": self.n_layers * self.mhc_params(),
            "embeddings": self.vocab * self.hidden,
            "mtp": self.mtp_params(),
            "vision": self.vision_params,
        }

    def mtp_params(self) -> int:
        """Multi-token-prediction draft layers: roughly one expert block each."""
        return self.mtp_layers * (self.routed_expert_params() + self.shared_expert_params())

    def total_params(self, include_vision: bool = True) -> int:
        s = self.params_summary()
        if not include_vision:
            s = {k: v for k, v in s.items() if k != "vision"}
        return sum(s.values())

    def active_params(self, include_vision: bool = False) -> int:
        """Parameters touched by one decode step for one token.

        `top_k` of the routed experts per layer, all shared/attention/router/emb
        weights, and the residual path. Vision is excluded from the text path.
        """
        s = self.params_summary()
        routed_active = s["routed_experts"] * self.top_k // self.n_experts
        return (
            routed_active
            + s["shared_experts"]
            + s.get("dense_ffn", 0)
            + s["attention_kda"]
            + s["attention_mla"]
            + s["attention_full"]
            + s["router"]
            + s["residual_mhc"]
            + s["embeddings"]
            + (s["vision"] if include_vision else 0)
        )

    def kv_bytes(self, context_tokens: int, bytes_per_param: float = 2.0) -> int:
        """KV cache for a given context: constant KDA state + linear MLA KV."""
        state = self.kda_layers * self.kda.state_bytes(self.mla.n_heads, bytes_per_param)
        mla_kv = self.mla_layers * self.mla.kv_bytes_per_token(bytes_per_param) * context_tokens
        full_kv = self.full_layers * self.full.kv_bytes_per_token(bytes_per_param) * context_tokens
        return state + mla_kv + full_kv

    def calibrate(self, stated_total: int, stated_active: int) -> dict:
        """Compare the derived model against the published headline numbers."""
        total = self.total_params(include_vision=False)
        active = self.active_params()
        return {
            "total": total,
            "total_stated": stated_total,
            "total_error": (total - stated_total) / stated_total,
            "active": active,
            "active_stated": stated_active,
            "active_error": (active - stated_active) / stated_active,
        }

    # -- config ------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str) -> "Architecture":
        import tomllib

        with open(path, "rb") as f:
            d = tomllib.load(f)
        mla_d = d.get("mla", {})
        kda_d = d.get("kda", {})
        full_d = d.get("full", {})
        leading_dense = int(d.get("leading_dense_layers", 0))
        dense_inter = int(d.get("dense_inter", 0))
        n_layers = int(d["n_layers"])
        # Both mistakes are silent in the model (a dense block modelled as routed
        # inflates the expert pool; a trace then fails to validate), so reject them
        # where the profile is read.
        if not 0 <= leading_dense <= n_layers:
            raise ValueError(
                f"leading_dense_layers({leading_dense}) must be between 0 and "
                f"n_layers({n_layers})"
            )
        if leading_dense and dense_inter <= 0:
            raise ValueError(
                f"leading_dense_layers({leading_dense}) needs dense_inter > 0 "
                "(the dense block's FFN width), or its weights are uncounted"
            )
        return cls(
            name=d.get("name", "unnamed"),
            n_layers=n_layers,
            hidden=int(d["hidden"]),
            vocab=int(d["vocab"]),
            n_experts=int(d["n_experts"]),
            top_k=int(d["top_k"]),
            moe_inter=int(d["moe_inter"]),
            n_shared_experts=int(d.get("n_shared_experts", 1)),
            shared_inter=int(d.get("shared_inter", d["moe_inter"])),
            kda_layers=int(d.get("kda_layers", 0)),
            mla_layers=int(d.get("mla_layers", 0)),
            full_layers=int(d.get("full_layers", 0)),
            mhc_streams=int(d.get("mhc_streams", 4)),
            mtp_layers=int(d.get("mtp_layers", 1)),
            vision_params=int(d.get("vision_params", 0)),
            leading_dense_layers=leading_dense,
            dense_inter=dense_inter,
            stated_total=int(d.get("stated_total", 0)),
            stated_active=int(d.get("stated_active", 0)),
            mla=MLA(
                n_heads=int(mla_d.get("n_heads", 48)),
                qk_nope=int(mla_d.get("qk_nope", 128)),
                qk_rope=int(mla_d.get("qk_rope", 64)),
                v_head=int(mla_d.get("v_head", 128)),
                kv_lora=int(mla_d.get("kv_lora", 512)),
            ),
            kda=KDA(
                hidden_multiple=float(kda_d.get("hidden_multiple", 7.0)),
                state_width=int(kda_d.get("state_width", 128)),
                d_conv=int(kda_d.get("d_conv", 4)),
            ),
            full=Full(
                n_heads=int(full_d.get("n_heads", 64)),
                n_kv_heads=int(full_d.get("n_kv_heads", 8)),
                head_dim=int(full_d.get("head_dim", 128)),
            ),
        )
