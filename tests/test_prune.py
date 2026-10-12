import json
import os
import unittest

from vipera.prune import mass_at_fraction, mass_curve, plan_from_trace

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
