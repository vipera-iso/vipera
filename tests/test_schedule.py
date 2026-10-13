import os
import unittest

from vipera.arch import Architecture
from vipera.manifest import MIB, Target
from vipera.schedule import Optimizations, Quant, model_bytes, simulate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLM = os.path.join(REPO_ROOT, "arch", "glm-5.3-flash.toml")
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")
LLAMA = os.path.join(REPO_ROOT, "arch", "llama3-70b.toml")
LLAMA_HW = os.path.join(REPO_ROOT, "hw", "llama70b-pc.toml")
OLMOE = os.path.join(REPO_ROOT, "arch", "olmoe-1b-7b.toml")


def glm() -> Architecture:
    return Architecture.from_toml(GLM)


def gtx1650() -> Target:
    return Target.from_toml(HW)


def roomy() -> Target:
    """A fantasy machine with 64 GiB of RAM — the ceiling if nothing streamed."""
    return Target(
        "roomy", vram_bytes=4096 * MIB, ram_bytes=64 * 1024 * MIB,
        vram_headroom_bytes=0, ram_headroom_bytes=0,
        vram_bw_gbs=192.0, ram_bw_gbs=40.0, disk_bw_gbs=2.0, cpu_gflops=1000.0,
    )


class TestModelBytes(unittest.TestCase):
    def test_expert_size_scales_with_quant(self):
        a = glm()
        fp8 = model_bytes(a, Quant.fp8())
        q4 = model_bytes(a, Quant.q4_experts())
        self.assertAlmostEqual(fp8.expert_each, q4.expert_each * 2, delta=2)

    def test_resident_excludes_routed_experts(self):
        a = glm()
        mb = model_bytes(a, Quant.fp8())
        self.assertLess(mb.resident_total, mb.expert_pool)


class TestGtx1650Run(unittest.TestCase):
    def setUp(self):
        self.result = simulate(glm(), gtx1650(), quant=Quant.fp8(), tokens=4, warmup=1, seed=1)

    def test_resident_weights_fit(self):
        # The always-needed weights must fit or nothing can run at all.
        self.assertTrue(self.result.feasible)
        r = self.result.manifest["residency"]
        self.assertEqual(r["resident_spill_disk_bytes"], 0)
        self.assertGreater(r["resident_bytes"], 0)

    def test_experts_spill_to_disk(self):
        self.assertGreater(self.result.disk_bytes_per_token, 0)

    def test_speed_is_sub_one_token_per_second(self):
        # 18B active parameters through a 2 GB/s disk: this is the honest verdict.
        self.assertLess(self.result.tok_s, 1.0)
        self.assertGreater(self.result.tok_s, 0.0)

    def test_super_hybrid_layer_kinds_present(self):
        kinds = {l["kind"] for l in self.result.manifest["layers"]}
        self.assertEqual(kinds, {"kda", "mla"})

    def test_layer_breakdown_is_per_layer(self):
        self.assertEqual(len(self.result.manifest["layers"]), glm().n_layers)


class TestCacheBehaviour(unittest.TestCase):
    def test_roomy_machine_hits_more_than_the_laptop(self):
        a = glm()
        laptop = simulate(a, gtx1650(), quant=Quant.fp8(), tokens=4, warmup=1, seed=2)
        big = simulate(a, roomy(), quant=Quant.fp8(), tokens=4, warmup=1, seed=2)
        self.assertGreater(big.hit_rate, laptop.hit_rate)
        self.assertLess(big.disk_bytes_per_token, laptop.disk_bytes_per_token)

    def test_repeated_routing_becomes_all_hits(self):
        a = glm()
        # Every layer requests the same 8 experts on every token.
        trace = [[[0, 1, 2, 3, 4, 5, 6, 7] for _ in range(a.n_layers)] for _ in range(4)]
        result = simulate(a, roomy(), quant=Quant.fp8(), routing=trace, warmup=1)
        self.assertEqual(result.disk_bytes_per_token, 0)
        self.assertAlmostEqual(result.hit_rate, 1.0, places=6)
        self.assertEqual(result.manifest["expert_cache"]["routing_source"], "trace")

    def test_lfu_beats_lru_on_skewed_routing(self):
        # A Zipf-skewed router keeps re-selecting a hot set; frequency-based
        # eviction should retain it better than pure recency.
        a = glm()
        lru = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=6, warmup=1, cache_policy="lru")
        lfu = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=6, warmup=1, cache_policy="lfu")
        self.assertGreater(lfu.hit_rate, lru.hit_rate)
        self.assertGreaterEqual(lfu.tok_s, lru.tok_s)
        self.assertEqual(lfu.manifest["expert_cache"]["policy"], "lfu")

    def test_unknown_cache_policy_rejected(self):
        with self.assertRaises(ValueError):
            simulate(glm(), gtx1650(), quant=Quant.fp8(), tokens=2, warmup=1, cache_policy="nope")


class TestOptimizations(unittest.TestCase):
    def test_expert_pruning_shrinks_the_pool(self):
        a = glm()
        full = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1)
        pruned = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1,
                          optimizations=Optimizations(prune_experts=0.05))
        self.assertLess(pruned.manifest["residency"]["expert_pool_bytes"],
                        full.manifest["residency"]["expert_pool_bytes"])

    def test_expert_pruning_raises_hit_rate(self):
        a = glm()
        full = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1, seed=3)
        pruned = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1, seed=3,
                          optimizations=Optimizations(prune_experts=0.05))
        self.assertGreater(pruned.hit_rate, full.hit_rate)
        self.assertLess(pruned.disk_bytes_per_token, full.disk_bytes_per_token)

    def test_experts_never_pruned_below_top_k(self):
        # Asking to keep 0% must still leave top_k experts to route to.
        r = simulate(glm(), gtx1650(), quant=Quant.fp8(), tokens=2, warmup=1,
                     optimizations=Optimizations(prune_experts=0.0))
        self.assertGreaterEqual(r.manifest["architecture"]["n_experts"], glm().top_k)

    def test_draft_tokens_multiplies_throughput(self):
        a = glm()
        base = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1)
        spec = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1,
                        optimizations=Optimizations(draft_tokens=3.0))
        self.assertAlmostEqual(spec.tok_s, base.tok_s * 3.0, places=6)

    def test_layer_pruning_cuts_layers_and_bytes(self):
        a = glm()
        full = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1)
        cut = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1,
                       optimizations=Optimizations(keep_layers=25))
        self.assertEqual(cut.manifest["architecture"]["n_layers"], 25)
        self.assertEqual(len(cut.manifest["layers"]), 25)
        self.assertLess(cut.bytes_per_token, full.bytes_per_token)

    def test_keep_layers_beyond_the_model_is_clamped(self):
        # `sweep` hardcodes keep_layers=25; on a 16-layer model that used to leave
        # `kinds` shorter than `n_layers` and crash on `kinds[i]`.
        a = Architecture.from_toml(OLMOE)
        r = simulate(a, gtx1650(), quant=Quant.q4_experts(), tokens=2, warmup=1,
                     optimizations=Optimizations(keep_layers=25))
        self.assertEqual(r.manifest["architecture"]["n_layers"], a.n_layers)
        self.assertEqual(len(r.manifest["layers"]), a.n_layers)

    def test_kv_bits_shrink_the_kv_cache(self):
        a = glm()
        big = simulate(a, gtx1650(), quant=Quant.fp8(), tokens=2, warmup=1,
                       optimizations=Optimizations(kv_bits=16))
        small = simulate(a, gtx1650(), quant=Quant.fp8(), tokens=2, warmup=1,
                         optimizations=Optimizations(kv_bits=8))
        self.assertEqual(small.manifest["context"]["kv_bytes"] * 2,
                         big.manifest["context"]["kv_bytes"])

    def test_optimizations_are_recorded_in_the_manifest(self):
        opt = Optimizations(prune_experts=0.1, keep_layers=30, kv_bits=8, draft_tokens=2.0)
        m = simulate(glm(), gtx1650(), quant=Quant.q4_experts(), tokens=2, warmup=1,
                     optimizations=opt).manifest
        self.assertEqual(m["optimizations"], opt.as_dict())


class TestDenseInstance(unittest.TestCase):
    """Llama-3-70B on the LLM-PC-OPT box (8 GB VRAM, 32 GB RAM, 3 GB/s SSD)."""

    def _run(self, **kw):
        return simulate(
            Architecture.from_toml(LLAMA), Target.from_toml(LLAMA_HW),
            quant=Quant.q4_experts(), context=4096, **kw,
        )

    def test_no_expert_pool_for_a_dense_model(self):
        r = self._run(tokens=2, warmup=1)
        self.assertEqual(r.manifest["residency"]["expert_pool_bytes"], 0)

    def test_ffn_counts_as_resident(self):
        r = self._run(tokens=2, warmup=1)
        # ~41 GB (decimal) of Q4 weights, all of them touched every token.
        self.assertGreater(r.manifest["residency"]["resident_bytes"], 40 * 10 ** 9)

    def test_baseline_is_far_below_the_five_tok_s_target(self):
        r = self._run(tokens=4, warmup=1)
        self.assertLess(r.tok_s, 2.0)
        self.assertGreater(r.tok_s, 0.0)

    def test_layer_pruning_is_the_only_dense_speedup(self):
        base = self._run(tokens=4, warmup=1)
        cut = self._run(tokens=4, warmup=1, optimizations=Optimizations(keep_layers=40))
        self.assertGreater(cut.tok_s, base.tok_s)
        self.assertLess(cut.bytes_per_token, base.bytes_per_token)

    def test_expert_pruning_is_a_no_op_for_dense(self):
        base = self._run(tokens=4, warmup=1)
        pruned = self._run(tokens=4, warmup=1, optimizations=Optimizations(prune_experts=0.05))
        self.assertEqual(pruned.tok_s, base.tok_s)


class TestAssumptions(unittest.TestCase):
    def test_prefetch_is_never_slower(self):
        a = glm()
        fast = simulate(a, gtx1650(), quant=Quant.fp8(), tokens=4, warmup=1, prefetch=True)
        slow = simulate(a, gtx1650(), quant=Quant.fp8(), tokens=4, warmup=1, prefetch=False)
        self.assertGreaterEqual(fast.tok_s, slow.tok_s)

    def test_deterministic_for_a_fixed_seed(self):
        a = glm()
        one = simulate(a, gtx1650(), quant=Quant.fp8(), tokens=4, warmup=1, seed=7)
        two = simulate(a, gtx1650(), quant=Quant.fp8(), tokens=4, warmup=1, seed=7)
        self.assertEqual(one.tok_s, two.tok_s)

    def test_infeasible_when_resident_exceeds_memory(self):
        tiny = Target(
            "tiny", vram_bytes=64 * MIB, ram_bytes=64 * MIB,
            vram_headroom_bytes=0, ram_headroom_bytes=0,
        )
        result = simulate(glm(), tiny, quant=Quant.fp8(), tokens=2, warmup=1)
        self.assertFalse(result.feasible)
        self.assertTrue(any("exceed" in w for w in result.warnings))

    def test_manifest_contract(self):
        m = simulate(glm(), gtx1650(), quant=Quant.fp8(), tokens=2, warmup=1).manifest
        for key in ("vipera_version", "architecture", "target", "quant", "residency",
                    "expert_cache", "performance", "layers", "context"):
            self.assertIn(key, m)
        self.assertEqual(m["expert_cache"]["policy"], "lru")


if __name__ == "__main__":
    unittest.main()
