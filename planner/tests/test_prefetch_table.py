import os
import tempfile
import unittest

from bench.prefetch_table import (
    build_table,
    format_table,
    n_experts_in,
    table_stats,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(REPO, "bench", "results", "olmoe-1b-7b-511tok.jsonl")


def parse_like_llama_cpp(text):
    """Mirror of `llama_moe_prefetch_table` in bench/llama-moe-prefetch.patch.

    Kept deliberately close to the C++ so a change to the writer that the patch
    would misread shows up here as a failing test.
    """
    table = {}
    for line in text.splitlines():
        p = line
        if p[:1] in ("#", "") or p.strip() == "":
            continue
        fields = p.split()
        if not fields or not fields[0].isdigit():
            continue
        il = int(fields[0])
        ids = []
        rest = p[len(fields[0]):]
        for tok in rest.replace(":", " ").replace(",", " ").split():
            if tok.isdigit():
                ids.append(int(tok))
        if ids:
            table[il] = ids
    return table


class TestNExperts(unittest.TestCase):
    def test_counts_past_the_largest_id(self):
        self.assertEqual(n_experts_in([[[0, 3], [1, 2]]]), 4)

    def test_empty_trace_has_no_experts(self):
        self.assertEqual(n_experts_in([]), 0)


class TestBuildTable(unittest.TestCase):
    def test_concentrated_routing_keeps_only_the_hot_experts(self):
        trace = [[[0, 1, 1, 0]] * 3 for _ in range(20)]
        table = build_table(trace, target_mass=0.95)
        self.assertEqual(len(table), 3)
        for keep in table:
            self.assertEqual(sorted(keep), [0, 1])

    def test_keeps_are_ordered_most_likely_first(self):
        # expert 5 is used twice as often as expert 2, which is used twice as often as 9
        trace = [[[5, 5, 2, 2], [5, 5, 2, 2], [5, 5, 2, 2]]]
        trace += [[[5, 5, 2, 2], [5, 5, 2, 2], [5, 5, 2, 2]]]
        trace += [[[5, 5, 9, 9], [5, 5, 9, 9], [5, 5, 9, 9]]]
        table = build_table(trace, target_mass=1.0)
        self.assertEqual(table[0], [5, 2, 9])

    def test_top_caps_each_layer(self):
        trace = [[[0, 1, 2, 3]] * 2 for _ in range(10)]
        table = build_table(trace, target_mass=1.0, top=2)
        for keep in table:
            self.assertEqual(len(keep), 2)

    def test_empty_trace_gives_no_layers(self):
        self.assertEqual(build_table([]), [])

    def test_one_entry_per_layer_in_order(self):
        trace = [[[0], [1], [2]] for _ in range(5)]
        self.assertEqual(len(build_table(trace, target_mass=1.0)), 3)


class TestFormat(unittest.TestCase):
    def test_round_trips_through_the_patch_parser(self):
        table = [[5, 2, 9], [], [0, 1]]
        parsed = parse_like_llama_cpp(format_table(table))
        self.assertEqual(parsed[0], [5, 2, 9])
        self.assertEqual(parsed[2], [0, 1])
        # the empty layer is omitted, never written as a blank candidate line
        self.assertNotIn(1, parsed)

    def test_header_lines_are_comments(self):
        text = format_table([[1]], source="bench/results/x.jsonl", tokens=7)
        for line in text.splitlines():
            if line.strip() and not line[0].isdigit():
                self.assertTrue(line.startswith("#"), line)

    def test_stats_count_every_candidate(self):
        self.assertEqual(table_stats([[1, 2], [3]]),
                         {"layers": 2, "experts": 3, "mean_per_layer": 1.5,
                          "max_per_layer": 2})

    def test_stats_of_an_empty_table(self):
        self.assertEqual(table_stats([])["experts"], 0)


@unittest.skipUnless(os.path.exists(REAL), "trace fixture not captured")
class TestRealTrace(unittest.TestCase):
    def test_table_covers_every_moe_layer(self):
        from vipera.trace import load_trace

        routing = load_trace(REAL, limit=256)
        table = build_table(routing, target_mass=0.95)
        self.assertEqual(len(table), 16)  # OLMoE-1B-7B has 16 layers
        for keep in table:
            self.assertTrue(keep)
        # a 0.95-mass keep-set must be far smaller than the 64 experts of a layer,
        # which is the whole reason a prefetch table fits in a small cache budget
        self.assertLess(table_stats(table)["max_per_layer"], 64)

    def test_real_table_parses_back(self):
        from vipera.trace import load_trace

        routing = load_trace(REAL, limit=128)
        table = build_table(routing, target_mass=0.95)
        parsed = parse_like_llama_cpp(format_table(table, source=REAL))
        for il, keep in enumerate(table):
            self.assertEqual(parsed.get(il), keep)


class TestMain(unittest.TestCase):
    def test_writes_a_file_the_patch_can_read(self):
        from bench.prefetch_table import main

        with tempfile.TemporaryDirectory() as d:
            trace = os.path.join(d, "t.jsonl")
            with open(trace, "w") as f:
                for t in range(8):
                    f.write('{"t": %d, "layers": [[0, 1, 1], [2, 2, 3]]}\n' % t)
            out = os.path.join(d, "pf.txt")
            self.assertEqual(main([trace, "-o", out, "--target-mass", "1.0"]), 0)
            with open(out) as f:
                parsed = parse_like_llama_cpp(f.read())
            self.assertEqual(parsed[0], [1, 0])
            self.assertEqual(parsed[1], [2, 3])


if __name__ == "__main__":
    unittest.main()
