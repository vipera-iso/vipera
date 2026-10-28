import os
import tempfile
import unittest

from tests._fixture import expected_bytes, make_gguf
from vipera.analyze import analyze, is_expert_tensor, is_router, n_experts_of
from vipera.gguf import read_gguf


class TestTensorClassifiers(unittest.TestCase):
    def test_expert_detection(self):
        self.assertTrue(is_expert_tensor("blk.3.ffn_gate_exps.weight"))
        self.assertFalse(is_expert_tensor("blk.3.ffn_gate_shexp.weight"))

    def test_router_detection(self):
        self.assertTrue(is_router("blk.3.ffn_gate_inp.weight"))
        self.assertFalse(is_router("blk.3.ffn_gate_exps.weight"))

    def test_dense_model_has_no_experts(self):
        self.assertFalse(is_expert_tensor("blk.0.ffn_gate.weight"))


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.gguf")
        make_gguf(self.path, block_count=2, n_experts=4)
        self.inv = analyze(read_gguf(self.path))
        self.exp = expected_bytes(block_count=2, n_experts=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts(self):
        self.assertEqual(self.inv.n_experts, 4)
        self.assertEqual(self.inv.n_layers, 2)

    def test_buckets(self):
        self.assertEqual(self.inv.resident_bytes, self.exp["resident"])
        self.assertEqual(self.inv.router_bytes, self.exp["router"])
        self.assertEqual(self.inv.shared_expert_bytes, self.exp["shared_expert"])
        self.assertEqual(self.inv.other_bytes, self.exp["other"])
        self.assertEqual(self.inv.expert_bytes, self.exp["expert"])

    def test_always_resident_excludes_experts(self):
        self.assertEqual(
            self.inv.always_resident_bytes,
            self.exp["resident"] + self.exp["router"] + self.exp["shared_expert"] + self.exp["other"],
        )
        self.assertEqual(self.inv.total_bytes, self.inv.always_resident_bytes + self.inv.expert_bytes)

    def test_per_layer_expert_split(self):
        per_layer = self.exp["expert"] // 2
        by_index = {l.index: l for l in self.inv.layers}
        self.assertEqual(by_index[0].expert_bytes, per_layer)
        self.assertEqual(by_index[1].expert_bytes, per_layer)
        self.assertEqual(by_index[0].n_experts, 4)

    def test_n_experts_of_requires_three_dims(self):
        g = read_gguf(self.path)
        self.assertEqual(n_experts_of(g.tensor("blk.0.ffn_gate_exps.weight")), 4)
        self.assertIsNone(n_experts_of(g.tensor("blk.0.attn_q.weight")))


if __name__ == "__main__":
    unittest.main()
