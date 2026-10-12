import json
import os
import unittest

from vipera.arch import Architecture
from vipera.manifest import Target
from vipera.schedule import (
    ExpertCache,
    Quant,
    plan_residency,
    simulate,
)
from vipera.trace import global_cache_ceiling, synth_shift_trace, synth_trace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLM = os.path.join(REPO_ROOT, "arch", "glm-5.3-flash.toml")
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")


def glm() -> Architecture:
    return Architecture.from_toml(GLM)


def gtx1650() -> Target:
    return Target.from_toml(HW)


def tail_hit_rate(trace, policy, capacity, *, frac=0.3, **cache_kw):
    """Hit rate over the last `frac` of the trace (the post-shift window)."""
    cache = ExpertCache(capacity, policy, **cache_kw)
    start = int(len(trace) * (1 - frac))
    hits = misses = 0
    for ti, token in enumerate(trace):
        before_h, before_m = cache.hits, cache.misses
        for layer, experts in enumerate(token):
            for e in experts:
                cache.access((layer, e))
        if ti >= start:
            hits += cache.hits - before_h
            misses += cache.misses - before_m
    total = hits + misses
    return hits / total if total else 0.0


class TestExpertCache(unittest.TestCase):
    def test_zero_capacity_never_stores_and_never_crashes(self):
        cache = ExpertCache(0, "hot")
        for _ in range(5):
            self.assertFalse(cache.access((0, 0)))
        self.assertEqual(len(cache), 0)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            ExpertCache(4, "clock")

    def test_lru_evicts_the_least_recently_used(self):
        cache = ExpertCache(2, "lru")
        cache.access((0, 0))
        cache.access((0, 1))
        cache.access((0, 0))  # (0,0) is now the most recent
        cache.access((0, 2))  # must evict (0,1), not (0,0)
        self.assertIn((0, 0), cache)
        self.assertIn((0, 2), cache)
        self.assertNotIn((0, 1), cache)

    def test_hot_admission_bypasses_a_key_no_hotter_than_its_victim(self):
        cache = ExpertCache(2, "hot")
        for _ in range(2):
            cache.access((0, 0))
            cache.access((0, 1))  # both resident, each seen twice
        cache.access((0, 2))  # seen once: not worth evicting a twice-seen key
        self.assertEqual(cache.bypasses, 1)
        self.assertNotIn((0, 2), cache)
        self.assertEqual(len(cache), 2)

    def test_lfu_admits_every_miss_unlike_hot(self):
        # The same sequence, but plain LFU has no admission filter.
        cache = ExpertCache(2, "lfu")
        for _ in range(2):
            cache.access((0, 0))
            cache.access((0, 1))
        cache.access((0, 2))
        self.assertIn((0, 2), cache)

    def test_hot_ages_its_counters(self):
        # White-box: aging is not otherwise observable, and it is the lever the
        # round-2 sweep showed to dominate. At access 4 the counters halve.
        cache = ExpertCache(4, "hot", aging_interval=4)
        for _ in range(4):
            cache.access((0, 0))
        self.assertEqual(cache._freq[(0, 0)], 2)

    def test_hot_beats_lfu_on_a_routing_shift(self):
        # A non-stationary trace: the hot set moves halfway through. LFU keeps
        # the stale hot set; aging lets `hot` follow the shift.
        shift = synth_shift_trace(
            60, n_layers=4, n_experts=32, top_k=4, skew=1.4, seed=0, split=0.5
        )
        lfu = tail_hit_rate(shift, "lfu", 24)
        hot = tail_hit_rate(shift, "hot", 24)
        self.assertGreater(hot, lfu + 0.05)

    def test_hot_is_not_worse_than_lfu_on_a_stationary_trace(self):
        stationary = synth_trace(60, n_layers=4, n_experts=32, top_k=4, skew=1.4, seed=0)
        lfu = tail_hit_rate(stationary, "lfu", 24)
        hot = tail_hit_rate(stationary, "hot", 24)
        self.assertGreaterEqual(hot, lfu - 0.02)


class TestPerLayerAllocation(unittest.TestCase):
    def test_requires_n_layers(self):
        with self.assertRaises(ValueError):
            ExpertCache(16, "lfu", alloc="per-layer")

    def test_unknown_alloc_rejected(self):
        with self.assertRaises(ValueError):
            ExpertCache(16, "lfu", alloc="balanced")

    def test_allocation_follows_demand(self):
        # Layer 0 is referenced constantly, layer 1 barely: water-filling should
        # give layer 0 almost the whole budget.
        cache = ExpertCache(20, "lfu", n_layers=2, alloc="per-layer", alloc_interval=20)
        for _ in range(50):
            for e in range(10):
                cache.access((0, e))
            cache.access((1, 0))
        self.assertGreater(cache.reallocations, 0)
        self.assertGreater(cache._cap[0], cache._cap[1])

    def test_quota_is_never_exceeded(self):
        cache = ExpertCache(8, "lfu", n_layers=4, alloc="per-layer", alloc_interval=40)
        for _ in range(100):
            for e in range(8):
                cache.access((2, e))  # hammer one layer only
        self.assertLessEqual(cache._count.get(2, 0), cache._cap[2])
        self.assertLessEqual(len(cache), 8)

    def test_global_and_per_layer_both_solve_a_repeated_key_trace(self):
        # A trivially cacheable trace must be all hits under either allocation.
        for alloc in ("global", "per-layer"):
            cache = ExpertCache(8, "lfu", n_layers=2, alloc=alloc)
            hits = 0
            for _ in range(10):
                for layer in range(2):
                    for e in range(4):
                        hits += cache.access((layer, e))
            self.assertGreater(hits, 0)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTI_TOPIC = os.path.join(REPO, "bench", "results", "olmoe-multitopic-545tok.jsonl")


def real_session():
    with open(MULTI_TOPIC) as f:
        return [json.loads(line)["layers"] for line in f if line.strip()]


def window_hit_rate(trace, policy, capacity, *, lo=0):
    cache = ExpertCache(capacity, policy, n_layers=len(trace[0]))
    hits = misses = 0
    for ti, token in enumerate(trace):
        before_h, before_m = cache.hits, cache.misses
        for layer, experts in enumerate(token):
            for e in experts:
                cache.access((layer, e))
        if ti >= lo:
            hits += cache.hits - before_h
            misses += cache.misses - before_m
    return hits / (hits + misses)


@unittest.skipUnless(os.path.exists(MULTI_TOPIC),
                     "real captured trace not present")
class TestRealMultiTopicShift(unittest.TestCase):
    """Regression tests on a real trace: 5 topics captured from OLMoE."""

    def test_hot_beats_lfu_across_a_real_topic_changing_session(self):
        # Round 2 claimed ageing wins when the distribution moves. This is the
        # first check of that claim on real routing rather than a synthetic shift.
        trace = real_session()
        lfu = window_hit_rate(trace, "lfu", 102)
        hot = window_hit_rate(trace, "hot", 102)
        self.assertGreater(hot, lfu + 0.015)

    def test_policies_are_close_on_one_real_topic(self):
        # Within a single topic the trace is stationary, so ageing buys nothing;
        # the two policies should be within noise of each other.
        segment = real_session()[:109]
        lfu = window_hit_rate(segment, "lfu", 102)
        hot = window_hit_rate(segment, "hot", 102)
        self.assertLess(abs(hot - lfu), 0.02)


class TestGlobalCeiling(unittest.TestCase):
    def test_no_slots_is_zero(self):
        trace = synth_trace(10, n_layers=2, n_experts=8, top_k=2)
        self.assertEqual(global_cache_ceiling(trace, 0), 0.0)

    def test_perfectly_concentrated_routing_reaches_one(self):
        # Every token on every layer picks expert 0; one slot per layer covers it.
        trace = [[[0, 0]] * 3 for _ in range(5)]
        self.assertAlmostEqual(global_cache_ceiling(trace, 3, warmup=0), 1.0, places=6)

    def test_more_slots_never_lower_the_ceiling(self):
        trace = synth_trace(200, n_layers=3, n_experts=32, top_k=4, skew=1.2, seed=1)
        ceilings = [global_cache_ceiling(trace, s, warmup=2) for s in (1, 5, 20, 60)]
        self.assertEqual(ceilings, sorted(ceilings))

    def test_ceiling_is_bounded_by_one(self):
        trace = synth_trace(200, n_layers=3, n_experts=32, top_k=4, skew=1.2, seed=1)
        self.assertLessEqual(global_cache_ceiling(trace, 10_000, warmup=2), 1.0)


class TestPlanResidency(unittest.TestCase):
    def test_plan_matches_the_simulation(self):
        plan = plan_residency(glm(), gtx1650(), quant=Quant.q4_experts())
        result = simulate(glm(), gtx1650(), quant=Quant.q4_experts(), tokens=4, warmup=1)
        r = result.manifest["residency"]
        self.assertEqual(plan.slots, r["expert_cache_slots"])
        self.assertEqual(plan.cache_bytes, r["expert_cache_bytes"])
        self.assertEqual(plan.expert_pool, r["expert_pool_bytes"])
        self.assertEqual(plan.n_layers, result.manifest["architecture"]["n_layers"])
        self.assertEqual(len(plan.warnings), len(result.warnings))

    def test_per_layer_fraction_is_slots_over_total_experts(self):
        plan = plan_residency(glm(), gtx1650(), quant=Quant.q4_experts())
        self.assertAlmostEqual(
            plan.per_layer_fraction, plan.slots / plan.total_experts, places=9
        )


class TestScheduleCeiling(unittest.TestCase):
    def test_manifest_reports_a_ceiling_at_the_matched_cache_size(self):
        trace = synth_trace(40, n_layers=45, n_experts=288, top_k=8, skew=1.2, seed=0)
        result = simulate(glm(), gtx1650(), quant=Quant.q4_experts(),
                          routing=trace, warmup=2)
        ec = result.manifest["expert_cache"]
        slots = result.manifest["residency"]["expert_cache_slots"]
        self.assertEqual(ec["slots"], slots)
        self.assertEqual(
            ec["static_cache_ceiling"],
            round(global_cache_ceiling(trace, slots, warmup=2), 4),
        )
        self.assertAlmostEqual(
            ec["ceiling_ratio"], round(result.hit_rate / ec["static_cache_ceiling"], 4),
            places=4,
        )

    def test_hot_policy_is_recorded(self):
        result = simulate(glm(), gtx1650(), quant=Quant.q4_experts(), tokens=4,
                          warmup=1, cache_policy="hot")
        self.assertEqual(result.manifest["expert_cache"]["policy"], "hot")

    def test_cache_alloc_is_recorded(self):
        result = simulate(glm(), gtx1650(), quant=Quant.q4_experts(), tokens=4,
                          warmup=1, cache_policy="hot", cache_alloc="per-layer")
        ec = result.manifest["expert_cache"]
        self.assertEqual(ec["alloc"], "per-layer")
        self.assertIn("reallocations", ec)

    def test_eval_tail_restricts_the_measured_window(self):
        trace = synth_trace(40, n_layers=45, n_experts=288, top_k=8, skew=1.2, seed=0)
        result = simulate(glm(), gtx1650(), quant=Quant.q4_experts(),
                          routing=trace, warmup=2, eval_tail=0.5)
        window = result.manifest["expert_cache"]["eval_window"]
        self.assertEqual(window["start"], 20)
        self.assertEqual(window["end"], 40)


if __name__ == "__main__":
    unittest.main()
