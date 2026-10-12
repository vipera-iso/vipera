import os
import unittest

from vipera.arch import Architecture
from vipera.manifest import MIB, Target
from vipera.schedule import Quant, model_bytes, simulate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLM = os.path.join(REPO_ROOT, "arch", "glm-5.3-flash.toml")
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")


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
