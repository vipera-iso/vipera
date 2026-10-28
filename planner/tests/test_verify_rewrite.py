import os
import tempfile
import unittest

from bench.verify_rewrite import compare
from tests._fixture import make_gguf_with_expert_data
from vipera.gguf_write import rewrite_experts


class VerifyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "src.gguf")
        make_gguf_with_expert_data(self.src, block_count=2, n_experts=4)

    def dest(self, name):
        return os.path.join(self.tmp.name, name)


class TestCompare(VerifyCase):
    def test_an_identity_rewrite_compares_equal(self):
        out = self.dest("same.gguf")
        rewrite_experts(self.src, out, {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]})
        problems, compared = compare(self.src, out)
        self.assertEqual(problems, [])
        self.assertEqual(compared, 9)

    def test_a_permutation_is_reported_with_the_tensor_names(self):
        out = self.dest("perm.gguf")
        rewrite_experts(self.src, out, {0: [3, 2, 1, 0], 1: [0, 1, 2, 3]})
        problems, _ = compare(self.src, out)
        self.assertEqual(len(problems), 1)
        self.assertIn("1 tensor(s) differ", problems[0])
        self.assertIn("blk.0.ffn_gate_exps.weight", problems[0])

    def test_a_permuted_and_unpermuted_round_trip_is_lossless(self):
        # this is the check that proves losslessness: forward then inverse
        shuffled, back = self.dest("shuffled.gguf"), self.dest("back.gguf")
        forward = {0: [2, 0, 3, 1], 1: [1, 3, 0, 2]}
        inverse = {}
        for layer, order in forward.items():
            inv = [0] * len(order)
            for slot, expert in enumerate(order):
                inv[expert] = slot
            inverse[layer] = inv
        rewrite_experts(self.src, shuffled, forward)
        rewrite_experts(shuffled, back, inverse)
        problems, compared = compare(self.src, back)
        self.assertEqual(problems, [])
        self.assertEqual(compared, 9)

    def test_a_prune_is_reported_as_a_descriptor_change(self):
        out = self.dest("pruned.gguf")
        rewrite_experts(self.src, out, {0: [0, 1], 1: [0, 1]})
        problems, _ = compare(self.src, out)
        self.assertTrue(any("descriptor changed" in p for p in problems))
        # and it must not pretend to have compared per-tensor digests
        self.assertEqual(problems[0].split()[0], "descriptor")


if __name__ == "__main__":
    unittest.main()
