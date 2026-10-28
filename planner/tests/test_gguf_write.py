import os
import tempfile
import unittest

from vipera.analyze import analyze
from vipera.gguf import GGUFError, read_gguf
from vipera.gguf_write import expert_order, expert_slab_bytes, rewrite_experts
from tests._fixture import make_gguf_with_expert_data


class WriterCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "src.gguf")
        self.patterns = make_gguf_with_expert_data(self.src, block_count=2, n_experts=4)
        self.slab = 64 * 64 * 4  # n_embd * n_ff * 4 bytes (F32)

    def dest(self, name="dest.gguf"):
        return os.path.join(self.tmp.name, name)

    def slab_of(self, path, layer, index, n_experts_in_file):
        """Bytes of expert `index` of layer `layer`, read back from a written file."""
        g = read_gguf(path)
        t = g.tensor(f"blk.{layer}.ffn_gate_exps.weight")
        slab = t.n_bytes // n_experts_in_file
        with open(path, "rb") as f:
            f.seek(g.data_offset + t.offset + index * slab)
            return f.read(slab)

    def assert_slab_holds(self, path, layer, position, expert, n_experts_in_file):
        """Slab `position` must hold `expert`'s own pattern - not merely match the source.

        Comparing only against the source would pass for a writer that emitted all
        zeros, so the fixture's per-(layer, expert) byte is checked directly.
        """
        want = self.patterns[(layer, expert)]
        got = self.slab_of(path, layer, position, n_experts_in_file)
        self.assertEqual(set(got), {want},
                         f"blk.{layer} slot {position} should hold expert {expert} "
                         f"(byte {want:#04x})")
        self.assertNotEqual(want, 0)


class TestSlabGeometry(WriterCase):
    def test_slab_is_the_expert_dimension_split(self):
        g = read_gguf(self.src)
        t = g.tensor("blk.0.ffn_gate_exps.weight")
        self.assertEqual(expert_slab_bytes(t, 4), self.slab)

    def test_a_tensor_without_an_expert_dimension_is_refused(self):
        g = read_gguf(self.src)
        with self.assertRaises(GGUFError):
            expert_slab_bytes(g.tensor("blk.0.attn_q.weight"), 4)


class TestRoundTrip(WriterCase):
    def test_copying_every_expert_unchanged_reproduces_the_file(self):
        order = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
        report = rewrite_experts(self.src, self.dest(), order)
        self.assertEqual(report.edited_tensors, 2)
        self.assertEqual(report.experts_before, report.experts_after)
        self.assertEqual(report.bytes_before, report.bytes_after)

        before, after = read_gguf(self.src), read_gguf(self.dest())
        self.assertEqual([t.name for t in before.tensors], [t.name for t in after.tensors])
        self.assertEqual([t.shape for t in before.tensors], [t.shape for t in after.tensors])
        for layer in (0, 1):
            for e in range(4):
                self.assert_slab_holds(self.dest(), layer, e, e, 4)
                self.assertEqual(self.slab_of(self.src, layer, e, 4),
                                 self.slab_of(self.dest(), layer, e, 4))

    def test_dense_tensors_are_copied_through_untouched(self):
        rewrite_experts(self.src, self.dest(), {0: [3, 2, 1, 0], 1: [0, 1, 2, 3]})
        g_before, g_after = read_gguf(self.src), read_gguf(self.dest())
        for name in ("token_embd.weight", "blk.0.attn_q.weight", "blk.1.ffn_gate_inp.weight",
                     "blk.0.ffn_gate_shexp.weight"):
            b, a = g_before.tensor(name), g_after.tensor(name)
            self.assertEqual(b.shape, a.shape, name)
            with open(self.src, "rb") as f:
                f.seek(g_before.data_offset + b.offset)
                raw_before = f.read(b.n_bytes)
            with open(self.dest(), "rb") as f:
                f.seek(g_after.data_offset + a.offset)
                raw_after = f.read(a.n_bytes)
            self.assertEqual(raw_before, raw_after, name)
            self.assertEqual(len(set(raw_before)), 1, f"{name} should hold one repeated byte")
            self.assertNotEqual(raw_before[0], 0, f"{name} should not be zeros")


class TestReorder(WriterCase):
    def test_slab_order_follows_the_permutation(self):
        # reverse the experts of layer 0; layer 1 untouched
        rewrite_experts(self.src, self.dest(), {0: [3, 2, 1, 0], 1: [0, 1, 2, 3]})
        dest = self.dest()
        self.assertEqual(read_gguf(dest).tensor("blk.0.ffn_gate_exps.weight").shape, (64, 64, 4))
        order = [3, 2, 1, 0]
        for new_index, old_index in enumerate(order):
            self.assert_slab_holds(dest, 0, new_index, old_index, 4)
            self.assertEqual(self.slab_of(dest, 0, new_index, 4),
                             self.slab_of(self.src, 0, old_index, 4),
                             f"expert slot {new_index} should hold expert {old_index}")

    def test_the_whole_file_still_validates_after_a_reorder(self):
        report = rewrite_experts(self.src, self.dest(), {0: [1, 3, 0, 2], 1: [2, 0, 3, 1]})
        g = read_gguf(self.dest())
        g.validate_extents()  # raises if a tensor points past the end of the file
        self.assertEqual(report.bytes_after, os.path.getsize(self.dest()))


class TestPrune(WriterCase):
    def test_expert_dimension_shrinks_to_the_kept_slabs(self):
        report = rewrite_experts(self.src, self.dest(), {0: [2, 0], 1: [1]})
        dest = self.dest()
        self.assertEqual(report.experts_before, 8)
        self.assertEqual(report.experts_after, 3)
        self.assertEqual(report.per_layer, {0: 2, 1: 1})

        g = read_gguf(dest)
        self.assertEqual(g.tensor("blk.0.ffn_gate_exps.weight").shape, (64, 64, 2))
        self.assertEqual(g.tensor("blk.1.ffn_gate_exps.weight").shape, (64, 64, 1))
        self.assert_slab_holds(dest, 0, 0, 2, 2)
        self.assert_slab_holds(dest, 0, 1, 0, 2)
        self.assert_slab_holds(dest, 1, 0, 1, 1)
        self.assertEqual(self.slab_of(dest, 0, 0, 2), self.slab_of(self.src, 0, 2, 4))
        self.assertEqual(self.slab_of(dest, 0, 1, 2), self.slab_of(self.src, 0, 0, 4))
        self.assertEqual(self.slab_of(dest, 1, 0, 1), self.slab_of(self.src, 1, 1, 4))

    def test_the_written_file_is_smaller_and_still_analyzable(self):
        before = analyze(read_gguf(self.src))
        report = rewrite_experts(self.src, self.dest(), {0: [0, 1], 1: [0, 1]})
        after = analyze(read_gguf(self.dest()))
        self.assertEqual(after.n_experts, 2)
        self.assertEqual(after.expert_bytes, before.expert_bytes // 2)
        self.assertEqual(report.bytes_after,
                         report.bytes_before - before.expert_bytes // 2)


class TestRefusals(WriterCase):
    def test_an_expert_id_outside_the_tensor_is_refused(self):
        with self.assertRaises(GGUFError):
            rewrite_experts(self.src, self.dest(), {0: [0, 9]})

    def test_a_layer_with_no_expert_tensor_is_refused(self):
        with self.assertRaises(GGUFError):
            rewrite_experts(self.src, self.dest(), {7: [0, 1]})

    def test_an_empty_expert_list_is_refused(self):
        with self.assertRaises(GGUFError):
            rewrite_experts(self.src, self.dest(), {0: []})

    def test_a_repeated_expert_id_is_refused(self):
        with self.assertRaises(GGUFError):
            rewrite_experts(self.src, self.dest(), {0: [1, 1]})

    def test_a_bad_alignment_is_refused(self):
        with self.assertRaises(GGUFError):
            rewrite_experts(self.src, self.dest(), {0: [0]}, alignment=3)


class TestExpertOrder(unittest.TestCase):
    def test_ranks_by_how_often_the_trace_uses_an_expert(self):
        routing = [
            [[0, 1], [3]],
            [[0, 2], [3]],
            [[2], [3]],
        ]
        self.assertEqual(expert_order(routing, n_experts=4),
                         {0: [0, 2, 1, 3], 1: [3, 0, 1, 2]})

    def test_prune_to_caps_the_list(self):
        routing = [[[0, 1, 2, 3], [3, 2, 1, 0]]]
        self.assertEqual(expert_order(routing, n_experts=4, prune_to=2),
                         {0: [0, 1], 1: [0, 1]})

    def test_the_order_is_deterministic_for_unused_experts(self):
        # nothing is routed: every expert ties, so ids break the tie in order
        self.assertEqual(expert_order([[[0], [0]]], n_experts=4)[0], [0, 1, 2, 3])

    def test_an_out_of_range_expert_in_the_trace_is_refused(self):
        with self.assertRaises(ValueError):
            expert_order([[[9], [0]]], n_experts=4)

    def test_bad_prune_to_is_refused(self):
        with self.assertRaises(ValueError):
            expert_order([[[0], [0]]], n_experts=4, prune_to=5)


if __name__ == "__main__":
    unittest.main()
