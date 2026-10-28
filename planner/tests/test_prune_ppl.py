import unittest

from bench.prune_ppl import arms, fit_proxy, format_report, spearman


def _rows(*specs):
    """(arm, kept_mass, ppl, stderr) tuples -> the shape `fit_proxy` reads."""
    return [{"arm": a, "kept_mass": m, "ppl": ppl, "ppl_ratio": ppl / 10.0,
             "ppl_stderr": se} for a, m, ppl, se in specs]


class TestArms(unittest.TestCase):
    def setUp(self):
        # skewed routing: experts 0..7 are used far more often than the rest
        self.trace = [[[i % 8 for i in range(8)] for _ in range(4)] for _ in range(40)]

    def test_every_fraction_gets_a_frequency_and_a_uniform_arm(self):
        out = arms(self.trace, n_experts=64, fractions=[0.25, 0.5])
        self.assertEqual([a["arm"] for a in out],
                         ["freq@0.25", "uniform@0.25", "freq@0.5", "uniform@0.5"])
        for a in out:
            self.assertEqual(len(a["keep"]), 4)  # 4 MoE layers in the fixture
            self.assertTrue(all(len(ids) == a["keep_per_layer"] for ids in a["keep"].values()))

    def test_the_uniform_arm_keeps_exactly_the_size_share_of_mass(self):
        out = {a["arm"]: a for a in arms(self.trace, n_experts=64, fractions=[0.25])}
        self.assertAlmostEqual(out["uniform@0.25"]["kept_mass"], 16 / 64, places=6)

    def test_the_frequency_arm_beats_the_uniform_one_on_mass(self):
        # this is the whole reason to rank by trace; if it did not hold, the two
        # arms would be measuring nothing
        out = {a["arm"]: a for a in arms(self.trace, n_experts=64, fractions=[0.25])}
        self.assertGreater(out["freq@0.25"]["kept_mass"],
                           out["uniform@0.25"]["kept_mass"])

    def test_both_arms_have_the_same_size(self):
        out = arms(self.trace, n_experts=64, fractions=[0.5])
        self.assertEqual(out[0]["keep_per_layer"], out[1]["keep_per_layer"])


class TestFitProxy(unittest.TestCase):
    def test_a_mass_shaped_curve_is_predicted_within_the_noise(self):
        # PPL ratio grows exactly as exp(2 * lost mass); the fit should recover it
        import math
        rows = _rows(*[(f"a{i}", m, 10.0 * math.exp(2.0 * (1.0 - m)), 0.3)
                       for i, m in enumerate((0.95, 0.75, 0.5, 0.25))])
        fit = fit_proxy(rows)
        self.assertTrue(fit["ok"], fit)
        self.assertAlmostEqual(fit["b"], 2.0, places=2)

    def test_a_curve_mass_cannot_explain_is_reported_as_a_miss(self):
        # same kept mass, wildly different quality: mass is not what decides it
        rows = _rows(("cheap", 0.5, 10.0, 0.1), ("dear", 0.5, 60.0, 0.6),
                     ("mid", 0.9, 10.0, 0.1))
        fit = fit_proxy(rows)
        self.assertFalse(fit["ok"])
        self.assertIsNotNone(fit["worst_arm"])

    def test_the_criterion_is_stated_in_the_report(self):
        fit = fit_proxy(_rows(("a", 0.9, 11.0, 0.2), ("b", 0.5, 20.0, 0.4)))
        self.assertIn("standard errors", fit["criterion"])
        self.assertIn("kept_mass", fit["form"])

    def test_one_arm_is_not_enough_to_fit(self):
        self.assertFalse(fit_proxy(_rows(("only", 0.5, 12.0, 0.2)))["ok"])


class TestSpearman(unittest.TestCase):
    def test_more_mass_with_less_perplexity_is_minus_one(self):
        # the direction a working proxy has: the arm that keeps more routing
        # mass is the arm with lower perplexity
        self.assertAlmostEqual(spearman([0.9, 0.5, 0.2], [3.0, 5.0, 9.0]), -1.0, places=6)

    def test_a_positive_rank_correlation_flags_the_wrong_direction(self):
        self.assertAlmostEqual(spearman([0.2, 0.5, 0.9], [3.0, 5.0, 9.0]), 1.0, places=6)

    def test_a_single_point_has_no_rank_correlation(self):
        self.assertEqual(spearman([0.5], [3.0]), 0.0)


class TestFormatReport(unittest.TestCase):
    def _report(self, ppl=45.2, ok=True):
        return {
            "ok": ok, "corpus": "c.raw", "chunks": 20, "context": 512,
            "baseline": {"ok": True, "ppl": 8.6426, "ppl_stderr": 0.30373},
            "arms": [{"arm": "freq@0.5", "policy": "trace-frequency", "fraction": 0.5,
                      "keep_per_layer": 32, "kept_mass": 0.71, "ok": True, "ppl": ppl,
                      "ppl_stderr": 1.2, "ppl_ratio": ppl / 8.6426,
                      "ppl_delta_pct": round(100 * (ppl / 8.6426 - 1), 2), "sigmas": 12.0}],
            "fit": {"ok": False, "form": "log(ppl_ratio) = b * (1 - kept_mass)", "b": 1.0,
                    "criterion": "every leave-one-out prediction within 2 standard errors",
                    "arms": [{"arm": "freq@0.5", "predicted_ratio": 1.1,
                              "measured_ratio": ppl / 8.6426, "error_pct": 400.0,
                              "sigmas": 30.0, "within_2_sigma": False}],
                    "worst_arm": "freq@0.5", "worst_error_pct": 400.0},
            "spearman_mass_vs_ppl": 1.0,
        }

    def test_the_table_shows_size_mass_and_the_measured_gap(self):
        text = format_report(self._report())
        self.assertIn("baseline      PPL 8.6426", text)
        self.assertIn("freq@0.5", text)
        self.assertIn("71.0%", text)
        self.assertIn("+423.0%", text)

    def test_a_failed_proxy_says_so_in_words(self):
        text = format_report(self._report())
        self.assertIn("does NOT predict", text)
        self.assertIn("MISS", text)

    def test_a_failed_baseline_short_circuits(self):
        text = format_report({"baseline": {"ok": False, "error": "timeout"}, "arms": []})
        self.assertIn("baseline FAILED: timeout", text)


if __name__ == "__main__":
    unittest.main()
