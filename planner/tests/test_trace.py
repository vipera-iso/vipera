import json
import os
import tempfile
import unittest

from vipera.trace import (
    TraceError,
    load_trace,
    save_trace,
    stats,
    synth_trace,
    validate,
)


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "trace.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_then_load_is_identical(self):
        routing = synth_trace(5, n_layers=3, n_experts=8, top_k=2, seed=1)
        save_trace(self.path, routing)
        self.assertEqual(load_trace(self.path), routing)

    def test_limit_stops_early(self):
        save_trace(self.path, synth_trace(10, n_layers=2, n_experts=4, top_k=2))
        self.assertEqual(len(load_trace(self.path, limit=3)), 3)


class TestValidation(unittest.TestCase):
    def _trace(self):
        return synth_trace(2, n_layers=3, n_experts=8, top_k=2, seed=0)

    def test_valid_passes(self):
        validate(self._trace(), n_layers=3, n_experts=8, top_k=2)

    def test_wrong_layer_count(self):
        with self.assertRaises(TraceError):
            validate(self._trace(), n_layers=4, n_experts=8, top_k=2)

    def test_wrong_top_k(self):
        with self.assertRaises(TraceError):
            validate(self._trace(), n_layers=3, n_experts=8, top_k=3)

    def test_expert_out_of_range(self):
        bad = [[[0, 99], [1, 2], [3, 4]]]
        with self.assertRaises(TraceError):
            validate(bad, n_layers=3, n_experts=8)

    def test_malformed_json_line(self):
        path = os.path.join(tempfile.mkdtemp(), "bad.jsonl")
        with open(path, "w") as f:
            f.write("{not json}\n")
        with self.assertRaises(TraceError):
            load_trace(path)

    def test_missing_layers(self):
        path = os.path.join(tempfile.mkdtemp(), "nolayers.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"t": 0}) + "\n")
        with self.assertRaises(TraceError):
            load_trace(path)

    def test_empty_trace_rejected(self):
        path = os.path.join(tempfile.mkdtemp(), "empty.jsonl")
        open(path, "w").close()
        with self.assertRaises(TraceError):
            load_trace(path)


class TestStats(unittest.TestCase):
    def test_concentrated_routing_has_low_entropy_and_high_ceiling(self):
        # Every token picks experts 0..3; a 50% static cache covers everything.
        # Entropy is log2(4)/log2(8) = 2/3, i.e. below uniform.
        routing = [[[0, 1, 2, 3]] * 4 for _ in range(8)]
        s = stats(routing, n_experts=8, cache_fraction=0.5)
        self.assertLess(s.mean_entropy, 0.75)
        self.assertAlmostEqual(s.static_cache_hit_ceiling, 1.0, places=6)

    def test_uniform_routing_has_entropy_near_one(self):
        # Cycle through all experts so each is equally frequent.
        routing = [[[i % 8 for i in range(8)]] for _ in range(8)]
        s = stats(routing, n_experts=8, cache_fraction=0.25)
        self.assertGreater(s.mean_entropy, 0.9)
        # A 25% static cache can only cover ~25% of a uniform workload.
        self.assertLess(s.static_cache_hit_ceiling, 0.4)

    def test_skew_raises_static_ceiling(self):
        uniform = synth_trace(200, n_layers=2, n_experts=64, top_k=4, skew=0.0, seed=3)
        skewed = synth_trace(200, n_layers=2, n_experts=64, top_k=4, skew=1.5, seed=3)
        cu = stats(uniform, n_experts=64, cache_fraction=0.05).static_cache_hit_ceiling
        cs = stats(skewed, n_experts=64, cache_fraction=0.05).static_cache_hit_ceiling
        self.assertGreater(cs, cu)

    def test_stats_are_json_serialisable(self):
        s = stats(synth_trace(10, n_layers=2, n_experts=8, top_k=2), n_experts=8)
        json.dumps(s.as_dict())


if __name__ == "__main__":
    unittest.main()
