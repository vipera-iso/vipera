import unittest

from bench.perplexity import parse_ppl


class TestParsePpl(unittest.TestCase):
    def test_parses_the_final_estimate(self):
        text = ("I perplexity: 15.87 seconds per pass - ETA 1.32 minutes\n"
                "I Final estimate: PPL = 8.6426 +/- 0.30373\n")
        self.assertEqual(parse_ppl(text), (8.6426, 0.30373))

    def test_takes_the_last_estimate_when_repeated(self):
        text = ("Final estimate: PPL = 9.0 +/- 0.5\n"
                "Final estimate: PPL = 8.5 +/- 0.4\n")
        self.assertEqual(parse_ppl(text), (8.5, 0.4))

    def test_returns_none_when_absent(self):
        self.assertIsNone(parse_ppl("llama_perf: everything fine, no number here"))


if __name__ == "__main__":
    unittest.main()
