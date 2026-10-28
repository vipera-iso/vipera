import os
import tempfile
import unittest

from bench.sparsity import _columns, channel_stats, format_report, read_tensor_f32
from tests._fixture import make_imatrix_gguf
from vipera.gguf import GGUFError


class TestChannelStats(unittest.TestCase):
    def test_uniform_energy_is_fully_participating(self):
        stats = channel_stats([1.0] * 10)
        self.assertEqual(stats["dead_channels"], 0)
        self.assertAlmostEqual(stats["participation_ratio"], 10.0)
        self.assertAlmostEqual(stats["top_decile_share"], 0.1)

    def test_energy_in_one_channel_has_a_participation_of_one(self):
        stats = channel_stats([100.0] + [0.0] * 9)
        self.assertAlmostEqual(stats["participation_ratio"], 1.0)
        self.assertAlmostEqual(stats["top_decile_share"], 1.0)
        self.assertEqual(stats["nonzero_channels"], 1)

    def test_dead_channels_are_counted_against_the_median(self):
        # median is 1.0, so anything below 0.01 is dead
        stats = channel_stats([1.0, 1.0, 1.0, 1.0, 0.005, 0.005, 2.0, 3.0])
        self.assertEqual(stats["dead_channels"], 2)
        self.assertAlmostEqual(stats["dead_fraction"], 2 / 8)

    def test_zero_vector_is_refused_rather_than_reported(self):
        with self.assertRaises(ValueError):
            channel_stats([0.0] * 4)

    def test_report_states_it_is_not_per_token_sparsity(self):
        text = format_report(channel_stats([1.0, 2.0]), tensor="t", corpus_note="n")
        self.assertIn("not claimed here", text)
        self.assertIn("static", text)


class TestReadTensor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "imatrix.gguf")
        make_imatrix_gguf(self.path, tensors={"blk.0.ffn_down.weight.in_sum2": [1.5, 2.5, 3.5]})

    def test_reads_values_in_order(self):
        shape, values = read_tensor_f32(self.path, "blk.0.ffn_down.weight.in_sum2")
        self.assertEqual(shape, (3,))
        self.assertEqual(values, [1.5, 2.5, 3.5])

    def test_missing_tensor_is_an_error(self):
        with self.assertRaises(GGUFError):
            read_tensor_f32(self.path, "blk.9.ffn_down.weight.in_sum2")


class TestExpertColumns(unittest.TestCase):
    def test_a_two_dimensional_tensor_splits_into_contiguous_expert_blocks(self):
        # GGUF stores ne0 contiguously, so expert e is a block of n_in values:
        # [c0e0, c1e0, c2e0, c0e1, c1e1, c2e1]
        values = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        columns, kind = _columns(values, (3, 2))
        self.assertIn("contiguous", kind)
        self.assertEqual(columns[0], [1.0, 2.0, 3.0])
        self.assertEqual(columns[1], [10.0, 20.0, 30.0])

    def test_the_strided_reading_is_not_used(self):
        # the wrong reading of the same bytes would pair 1 with 3 rather than 1 with 2
        values = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        columns, _ = _columns(values, (3, 2))
        self.assertNotEqual(columns[0], [1.0, 10.0])

    def test_a_dense_tensor_is_one_column(self):
        columns, kind = _columns([1.0, 2.0, 3.0], (3,))
        self.assertEqual(kind, "all experts summed")
        self.assertEqual(columns, [[1.0, 2.0, 3.0]])

    def test_a_single_column_expert_tensor_is_not_split(self):
        columns, _ = _columns([1.0, 2.0], (2, 1))
        self.assertEqual(len(columns), 1)


if __name__ == "__main__":
    unittest.main()
