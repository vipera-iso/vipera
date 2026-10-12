import unittest

from vipera.scoring import local_llm_16gb_score


class TestLocalLLM16GBScore(unittest.TestCase):
    def test_formula_returns_one_at_every_target(self):
        # Q=100, T=10, M=16, P=65, C=32k -> every factor is 1.
        s = local_llm_16gb_score(quality_pct=100, tok_s=10, ram_gb=16, power_w=65, context=32768)
        self.assertAlmostEqual(s, 1.0, places=12)

    def test_known_partial_case(self):
        # Q=99, T=0.112, M=25.77, P=65, C=32768
        s = local_llm_16gb_score(
            quality_pct=99, tok_s=0.112, ram_gb=25.77, power_w=65, context=32768
        )
        self.assertAlmostEqual(s, 0.99 * 0.0112 * (16 / 25.77), places=6)

    def test_score_scales_with_tokens_per_second(self):
        lo = local_llm_16gb_score(quality_pct=100, tok_s=5, ram_gb=16, power_w=65, context=32768)
        hi = local_llm_16gb_score(quality_pct=100, tok_s=10, ram_gb=16, power_w=65, context=32768)
        self.assertAlmostEqual(hi, lo * 2, places=12)

    def test_longer_context_costs_score(self):
        a = local_llm_16gb_score(quality_pct=100, tok_s=10, ram_gb=16, power_w=65, context=16384)
        b = local_llm_16gb_score(quality_pct=100, tok_s=10, ram_gb=16, power_w=65, context=32768)
        self.assertGreater(a, b)

    def test_rejects_nonsense_inputs(self):
        with self.assertRaises(ValueError):
            local_llm_16gb_score(quality_pct=100, tok_s=10, ram_gb=0, power_w=65, context=32768)


if __name__ == "__main__":
    unittest.main()
