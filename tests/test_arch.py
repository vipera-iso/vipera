import os
import tempfile
import unittest

from vipera.arch import Architecture

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLM = os.path.join(REPO_ROOT, "arch", "glm-5.3-flash.toml")
LLAMA = os.path.join(REPO_ROOT, "arch", "llama3-70b.toml")
DEEPSEEK = os.path.join(REPO_ROOT, "arch", "deepseek-v2-lite.toml")


def profile(**overrides) -> str:
    """Write a minimal profile to a temp file; caller unlinks it."""
    body = dict(
        name="t", n_layers=4, hidden=16, vocab=16, n_experts=4, top_k=1,
        moe_inter=8, full_layers=4,
    )
    body.update(overrides)
    text = "\n".join(f"{k} = {v!r}" for k, v in body.items()) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(text)
    return f.name


def toy() -> Architecture:
    return Architecture(
        name="toy",
        n_layers=4,
        hidden=64,
        vocab=100,
        n_experts=8,
        top_k=2,
        moe_inter=32,
        n_shared_experts=1,
        shared_inter=32,
        kda_layers=3,
        mla_layers=1,
    )


class TestLayerKinds(unittest.TestCase):
    def test_counts_match_request(self):
        a = toy()
        kinds = a.layer_kinds()
        self.assertEqual(len(kinds), 4)
        self.assertEqual(kinds.count("kda"), 3)
        self.assertEqual(kinds.count("mla"), 1)

    def test_super_hybrid_cadence(self):
        a = Architecture(
            name="x", n_layers=8, hidden=16, vocab=16, n_experts=4, top_k=1, moe_inter=8,
            kda_layers=6, mla_layers=2,
        )
        # every 4th layer is MLA
        self.assertEqual(a.layer_kinds(), ["kda", "kda", "kda", "mla"] * 2)

    def test_mismatched_counts_rejected(self):
        a = Architecture(
            name="x", n_layers=4, hidden=16, vocab=16, n_experts=4, top_k=1, moe_inter=8,
            kda_layers=2, mla_layers=1,
        )
        with self.assertRaises(ValueError):
            a.layer_kinds()


class TestParams(unittest.TestCase):
    def test_expert_params(self):
        a = toy()
        self.assertEqual(a.expert_params_each, 3 * 64 * 32)
        self.assertEqual(a.routed_expert_params(), 8 * 3 * 64 * 32)

    def test_active_less_than_total(self):
        a = toy()
        self.assertLess(a.active_params(), a.total_params(include_vision=False))

    def test_kv_grows_with_context(self):
        a = toy()
        self.assertLess(a.kv_bytes(128), a.kv_bytes(4096))


class TestShippedProfile(unittest.TestCase):
    def test_profile_loads(self):
        a = Architecture.from_toml(GLM)
        self.assertEqual(a.n_layers, 45)
        self.assertEqual(a.n_experts, 288)
        self.assertEqual(a.top_k, 8)
        self.assertEqual(a.kda_layers + a.mla_layers, a.n_layers)

    def test_profile_calibrates_within_one_percent(self):
        a = Architecture.from_toml(GLM)
        cal = a.calibrate(a.stated_total, a.stated_active)
        self.assertLess(abs(cal["total_error"]), 0.01)
        self.assertLess(abs(cal["active_error"]), 0.01)

    def test_profile_is_a_45_layer_super_hybrid(self):
        kinds = Architecture.from_toml(GLM).layer_kinds()
        self.assertEqual(kinds.count("kda"), 34)
        self.assertEqual(kinds.count("mla"), 11)


class TestDenseProfile(unittest.TestCase):
    def test_llama_profile_is_dense(self):
        a = Architecture.from_toml(LLAMA)
        self.assertTrue(a.dense)
        self.assertEqual(a.n_experts, 1)

    def test_every_layer_is_full_attention(self):
        a = Architecture.from_toml(LLAMA)
        self.assertEqual(a.layer_kinds(), ["full"] * a.n_layers)

    def test_total_is_about_70b(self):
        a = Architecture.from_toml(LLAMA)
        total = a.total_params(include_vision=False)
        self.assertLess(abs(total - a.stated_total) / a.stated_total, 0.02)

    def test_dense_reads_all_parameters_every_token(self):
        # No sparsity: active == total for a dense model.
        a = Architecture.from_toml(LLAMA)
        self.assertEqual(a.active_params(), a.total_params(include_vision=False))

    def test_gqa_kv_shrinks_with_kv_heads(self):
        a = Architecture.from_toml(LLAMA)
        # 8 KV heads of 128 dims -> 2*8*128 = 2048 bytes/token/layer at fp16
        self.assertEqual(a.full.kv_bytes_per_token(2.0), 2 * 8 * 128 * 2)
        self.assertGreater(a.kv_bytes(4096), a.kv_bytes(1024))


class TestLeadingDenseBlocks(unittest.TestCase):
    """A dense leading block: no router, no experts, one always-read FFN."""

    def arch(self) -> Architecture:
        return Architecture(
            name="dl", n_layers=6, hidden=32, vocab=64, n_experts=4, top_k=1,
            moe_inter=16, n_shared_experts=1, shared_inter=16,
            kda_layers=0, mla_layers=0, full_layers=6,
            leading_dense_layers=2, dense_inter=96,
        )

    def test_moe_layers_is_the_trace_dimension(self):
        a = self.arch()
        self.assertEqual(a.n_layers, 6)
        self.assertEqual(a.moe_layers, 4)

    def test_dense_blocks_carry_no_router_or_experts(self):
        a = self.arch()
        s = a.params_summary()
        self.assertEqual(s["routed_experts"], 4 * a.routed_expert_params())
        self.assertEqual(s["shared_experts"], 4 * a.shared_expert_params())
        self.assertEqual(s["router"], 4 * a.router_params())
        self.assertEqual(s["dense_ffn"], 2 * 3 * 32 * 96)

    def test_dense_ffn_is_active_every_token(self):
        # The dense FFN is read in full; only top_k of each layer's experts are.
        a = self.arch()
        s = a.params_summary()
        self.assertEqual(
            a.active_params(),
            s["routed_experts"] * a.top_k // a.n_experts
            + s["shared_experts"] + s["dense_ffn"] + s["attention_full"]
            + s["router"] + s["residual_mhc"] + s["embeddings"],
        )

    def test_no_dense_block_is_exactly_the_old_arithmetic(self):
        a = self.arch()
        plain = Architecture(
            name=a.name, n_layers=a.n_layers, hidden=a.hidden, vocab=a.vocab,
            n_experts=a.n_experts, top_k=a.top_k, moe_inter=a.moe_inter,
            n_shared_experts=a.n_shared_experts, shared_inter=a.shared_inter,
            kda_layers=0, mla_layers=0, full_layers=a.full_layers,
        )
        self.assertEqual(plain.moe_layers, plain.n_layers)
        self.assertEqual(
            plain.params_summary()["routed_experts"],
            plain.n_layers * plain.routed_expert_params(),
        )
        self.assertNotIn("dense_ffn", plain.params_summary())

    def test_dense_block_without_a_width_is_rejected(self):
        path = profile(leading_dense_layers=1)
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError) as cm:
            Architecture.from_toml(path)
        self.assertIn("dense_inter", str(cm.exception))

    def test_dense_block_beyond_the_model_is_rejected(self):
        path = profile(leading_dense_layers=5, dense_inter=32)
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError):
            Architecture.from_toml(path)

    def test_an_acceptable_dense_block_loads(self):
        path = profile(leading_dense_layers=1, dense_inter=32)
        self.addCleanup(os.unlink, path)
        a = Architecture.from_toml(path)
        self.assertEqual(a.moe_layers, 3)
        self.assertEqual(a.dense_inter, 32)

    def test_shipped_deepseek_profile_expresses_its_dense_block(self):
        a = Architecture.from_toml(DEEPSEEK)
        self.assertEqual(a.n_layers, 27)
        self.assertEqual(a.leading_dense_layers, 1)
        self.assertEqual(a.moe_layers, 26)
        # 26 MoE blocks of 64 experts is what the trace carries and what it meters
        self.assertEqual(a.params_summary()["routed_experts"], 26 * a.routed_expert_params())
        cal = a.calibrate(a.stated_total, a.stated_active)
        self.assertLess(abs(cal["total_error"]), 0.02)
        self.assertLess(abs(cal["active_error"]), 0.03)


if __name__ == "__main__":
    unittest.main()
