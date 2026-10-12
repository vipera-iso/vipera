import os
import unittest

from vipera.arch import Architecture

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLM = os.path.join(REPO_ROOT, "arch", "glm-5.3-flash.toml")
LLAMA = os.path.join(REPO_ROOT, "arch", "llama3-70b.toml")


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


if __name__ == "__main__":
    unittest.main()
