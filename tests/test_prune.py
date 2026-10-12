import json
import os
import unittest

from vipera.prune import (
    gated_prune,
    mass_at_fraction,
    mass_curve,
    plan_from_trace,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(REPO, "bench", "results", "olmoe-1b-7b-511tok.jsonl")


class TestPlanFromTrace(unittest.TestCase):
    def test_concentrated_routing_needs_a_tiny_keep_set(self):
        # Every token uses only experts 0 and 1.
        trace = [[[0, 1, 1, 0]] * 3 for _ in range(20)]
        plan = plan_from_trace(trace, n_experts=16, target_mass=0.95)
        self.assertEqual(len(plan.layers), 3)
        for layer in plan.layers:
            self.assertEqual(sorted(layer.keep), [0, 1])
            self.assertAlmostEqual(layer.kept_mass, 1.0, places=6)
        self.assertLess(plan.keep_fraction, 0.2)

    def test_uniform_routing_needs_roughly_the_target_fraction(self):
        # Cycle experts so every one is equally used.
        trace = [[[i % 8 for i in range(8)]] for _ in range(8)]
        plan = plan_from_trace(trace, n_experts=8, target_mass=0.5)
        # Uniform mass: 50% needs ~4 of 8 experts, i.e. ~0.5 of the pool.
        self.assertAlmostEqual(plan.keep_fraction, 0.5, delta=0.15)

    def test_rejects_out_of_range_target(self):
        trace = [[[0, 1]]]
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                plan_from_trace(trace, n_experts=4, target_mass=bad)

    def test_report_is_json_serialisable(self):
        trace = [[[0, 1], [2, 3]] for _ in range(4)]
        json.dumps(plan_from_trace(trace, n_experts=4).as_dict())


class TestMassAtFraction(unittest.TestCase):
    def test_concentrated_routing_beats_uniform_by_a_lot(self):
        trace = [[[0, 0, 0, 0]] * 2 for _ in range(10)]  # always expert 0
        mass = mass_at_fraction(trace, n_experts=64, fraction=0.05)
        self.assertGreater(mass, 0.9)

    def test_uniform_routing_retains_about_the_fraction(self):
        trace = [[[i % 8 for i in range(8)]] for _ in range(8)]
        mass = mass_at_fraction(trace, n_experts=8, fraction=0.5)
        self.assertAlmostEqual(mass, 0.5, delta=0.1)

    def test_curve_pairs_each_fraction_with_the_uniform_baseline(self):
        trace = [[[0, 1]] * 2 for _ in range(5)]
        curve = mass_curve(trace, n_experts=4, fractions=[0.25, 0.5])
        self.assertEqual([r["fraction"] for r in curve], [0.25, 0.5])
        self.assertEqual([r["uniform_kept_mass"] for r in curve], [0.25, 0.5])


class TestGatedPrune(unittest.TestCase):
    def test_passes_the_gate_on_concentrated_routing(self):
        trace = [[[0, 1]] * 3 for _ in range(20)]
        plan = gated_prune(trace, n_experts=16, max_keep_fraction=0.5)
        self.assertEqual(plan.pruned_layers, 3)
        self.assertEqual(plan.rolled_back_layers, 0)
        for layer in plan.layers:
            self.assertTrue(layer.pruned)
            self.assertEqual(layer.reason, "pruned")
            self.assertEqual(sorted(layer.keep), [0, 1])

    def test_rolls_back_when_the_mass_keep_set_is_too_big(self):
        # Routing is confined to 4 of 8 experts, so 95% of the mass needs those 4
        # (50% of the layer) — over a 30% budget, so the layer is kept whole.
        trace = [[[i % 4 for i in range(8)]] for _ in range(8)]
        plan = gated_prune(trace, n_experts=8, target_mass=0.95, max_keep_fraction=0.3)
        self.assertEqual(plan.rolled_back_layers, 1)
        layer = plan.layers[0]
        self.assertFalse(layer.pruned)
        self.assertEqual(layer.reason, "mass_unreachable")
        self.assertEqual(layer.candidate_keep, 4)
        self.assertEqual(len(layer.keep), 8)
        self.assertAlmostEqual(layer.kept_mass, 1.0, places=6)

    def test_drift_gate_rolls_back_only_the_drifting_layer(self):
        trace = [[[0, 1]] * 2 for _ in range(10)]
        plan = gated_prune(trace, n_experts=16, delta=0.1, drift={0: 0.05, 1: 0.5})
        self.assertTrue(plan.layers[0].pruned)
        self.assertFalse(plan.layers[1].pruned)
        self.assertEqual(plan.layers[1].reason, "drift")
        self.assertEqual(plan.layers[1].drift, 0.5)

    def test_missing_drift_entry_is_treated_as_zero(self):
        trace = [[[0, 1]] * 2 for _ in range(10)]
        plan = gated_prune(trace, n_experts=16, delta=0.1, drift={})
        self.assertEqual(plan.pruned_layers, len(plan.layers))

    def test_nothing_to_prune_when_the_keep_set_is_the_whole_layer(self):
        trace = [[list(range(8))] for _ in range(8)]
        plan = gated_prune(trace, n_experts=8, target_mass=0.95)
        self.assertEqual(plan.layers[0].reason, "nothing_to_prune")
        self.assertFalse(plan.layers[0].pruned)

    def test_delta_without_drift_data_is_an_error(self):
        trace = [[[0, 1]]]
        with self.assertRaises(ValueError):
            gated_prune(trace, n_experts=4, delta=0.1, drift=None)

    def test_rejects_out_of_range_budgets(self):
        trace = [[[0, 1]]]
        for kwargs in ({"target_mass": 0.0}, {"max_keep_fraction": 0.0},
                       {"max_keep_fraction": 1.5}):
            with self.assertRaises(ValueError):
                gated_prune(trace, n_experts=4, **kwargs)

    def test_report_is_json_serialisable_and_counts_reasons(self):
        trace = [[[0, 1]] * 2 for _ in range(10)]
        data = gated_prune(trace, n_experts=16, max_keep_fraction=0.5).as_dict()
        json.dumps(data)
        self.assertEqual(data["algorithm"], "gated-expert-pruning")
        self.assertEqual(data["reasons"]["pruned"], len(data["layers"]))
        self.assertIn("rolled_back_layers", data)


@unittest.skipUnless(os.path.exists(REAL), "real captured trace not present")
class TestGatedPruneOnRealRouting(unittest.TestCase):
    def _trace(self):
        with open(REAL) as f:
            return [json.loads(l)["layers"] for l in f if l.strip()]

    def test_a_five_percent_budget_rolls_back_every_layer(self):
        # On real routing a 5% keep-set holds far less than 95% of the mass, so the
        # gate must refuse to prune rather than pretend the pool shrank for free.
        plan = gated_prune(self._trace(), n_experts=64, target_mass=0.95,
                           max_keep_fraction=0.05)
        self.assertEqual(plan.pruned_layers, 0)
        self.assertEqual(plan.rolled_back_layers, len(plan.layers))
        self.assertEqual(set(plan.as_dict()["reasons"]), {"mass_unreachable"})

    def test_a_generous_budget_prunes_every_layer(self):
        plan = gated_prune(self._trace(), n_experts=64, target_mass=0.95,
                           max_keep_fraction=1.0)
        self.assertEqual(plan.rolled_back_layers, 0)
        self.assertGreater(plan.pruned_layers, 0)


@unittest.skipUnless(os.path.exists(REAL), "real captured trace not present")
class TestAgainstRealRouting(unittest.TestCase):
    def test_pruning_to_five_percent_discards_most_routing_mass(self):
        # This is the claim the scheduler's `prune_experts` lever silently assumes
        # away: that keeping 5% of experts is (nearly) free.
        with open(REAL) as f:
            trace = [json.loads(l)["layers"] for l in f if l.strip()]
        mass = mass_at_fraction(trace, n_experts=64, fraction=0.05)
        self.assertLess(mass, 0.35)
        self.assertGreater(mass, 0.05)  # still far better than a random 5%


if __name__ == "__main__":
    unittest.main()
