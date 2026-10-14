import unittest

from bench.perplexity import (EVICTION_POLICIES, POLICY_ENV, build_command, compare_stream,
                              difference_sigmas, format_comparison, format_table, parse_ppl,
                              parse_stream_stats, run_perplexity)


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


class TestParseStreamStats(unittest.TestCase):
    def test_reads_shifts_and_scored_tokens(self):
        text = "I perplexity_stream: 13 context shifts, 5121 tokens scored, sinks = 4\n"
        self.assertEqual(parse_stream_stats(text), (5121, 13))

    def test_absent_on_the_stock_path(self):
        self.assertEqual(parse_stream_stats("computing over 20 chunks, n_ctx=768"), (None, None))


class TestBuildCommand(unittest.TestCase):
    def test_stock_run_has_no_stride_or_keep(self):
        cmd = build_command("ppl", "m.gguf", corpus="c.raw", context=512, threads=4)
        self.assertNotIn("--ppl-stride", cmd)
        self.assertNotIn("--keep", cmd)
        self.assertNotIn("--context-shift", cmd)
        self.assertNotIn("-b", cmd)

    def test_stride_pins_the_batch_to_the_context(self):
        # the tool derives n_parallel from n_batch/n_ctx and then cannot place a
        # full-size batch, so the harness must keep them equal.
        cmd = build_command("ppl", "m.gguf", corpus="c.raw", context=512, stride=256)
        self.assertEqual(cmd[cmd.index("-b") + 1], "512")
        self.assertEqual(cmd[cmd.index("--ppl-stride") + 1], "256")

    def test_streaming_adds_keep_and_context_shift(self):
        cmd = build_command("ppl", "m.gguf", corpus="c.raw", context=512, stride=256,
                            keep=4, context_shift=True)
        self.assertEqual(cmd[cmd.index("--keep") + 1], "4")
        self.assertIn("--context-shift", cmd)

    def test_keep_without_stride_is_still_forwarded(self):
        # a knob the tool ignores on its own is still the user's business
        cmd = build_command("ppl", "m.gguf", corpus="c.raw", keep=32)
        self.assertEqual(cmd[cmd.index("--keep") + 1], "32")


class TestEvictionPolicy(unittest.TestCase):
    def test_the_four_policies_are_offered(self):
        self.assertEqual(EVICTION_POLICIES,
                         ("recency", "recency-gap", "surprise", "surprise-local"))

    def test_the_policy_travels_by_environment_not_argv(self):
        # the patch reads it from the env, which is vipera's rule for llama.cpp patches
        self.assertEqual(POLICY_ENV, "LLAMA_PPL_STREAM_POLICY")
        cmd = build_command("ppl", "m.gguf", corpus="c.raw", stride=256, keep=4,
                            context_shift=True)
        self.assertNotIn(POLICY_ENV, cmd)
        self.assertNotIn("--policy", cmd)

    def test_an_unknown_policy_is_refused_before_spawning(self):
        # a typo must not silently fall back to recency and be reported as a result
        with self.assertRaises(ValueError):
            run_perplexity("definitely-not-a-binary", "m.gguf", corpus="c.raw", policy="turbo")

    def test_the_table_names_the_policy(self):
        report = {"ok": True, "exit_code": 0, "elapsed_s": 1.0, "corpus": "c.raw",
                  "chunks": 6, "context": 512, "keep": 4, "policy": "surprise",
                  "scored_tokens": 1533, "context_shifts": 4, "ppl": 19.49,
                  "ppl_stderr": 1.69}
        self.assertIn("policy surprise", format_table(report))
        # and a report from before policies existed still renders
        report.pop("policy")
        self.assertIn("policy recency", format_table(report))


class TestCompareStream(unittest.TestCase):
    def _fake(self, stock_ppl, stock_se, stream_ppl, stream_se):
        def run(binary, model, **kw):
            stream = bool(kw.get("keep"))
            return {"ok": True, "exit_code": 0, "elapsed_s": 1.0, "chunks": kw.get("chunks"),
                    "context": kw.get("context"), "corpus": "c.raw", "stride": kw.get("stride"),
                    "keep": kw.get("keep"), "context_shift": kw.get("context_shift"),
                    "ppl": stream_ppl if stream else stock_ppl,
                    "ppl_stderr": stream_se if stream else stock_se}
        return run

    def _compare(self, run):
        import bench.perplexity as mod
        original = mod.run_perplexity
        mod.run_perplexity = run
        try:
            return compare_stream("ppl", "m.gguf", corpus="c.raw", chunks=20,
                                  context=512, stride=256, keep=4)
        finally:
            mod.run_perplexity = original

    def test_reports_the_gap_when_it_clears_the_noise(self):
        report = self._compare(self._fake(23.46, 0.30, 30.00, 0.40))
        self.assertTrue(report["ok"])
        self.assertAlmostEqual(report["delta_ppl"], 6.54, places=6)
        self.assertAlmostEqual(report["delta_pct"], 27.87, places=1)
        self.assertAlmostEqual(report["difference_stderr"], 0.5, places=6)
        self.assertAlmostEqual(report["difference_sigmas"], 13.08, places=1)
        self.assertTrue(report["significant"])
        self.assertIn("larger than the noise", format_comparison(report))

    def test_the_screen_is_one_sigma_and_says_so(self):
        # a 1.5-sigma gap clears the screen but is not a 95% claim, and the report
        # must not let the reader infer one
        report = self._compare(self._fake(20.00, 0.50, 21.00, 0.50))
        self.assertAlmostEqual(report["difference_sigmas"], 1.41, places=2)
        self.assertTrue(report["significant"])
        text = format_comparison(report)
        self.assertIn("1.4 sigma", text)
        self.assertIn("95% claim needs 2", text)

    def test_calls_a_gap_inside_the_noise_noise(self):
        report = self._compare(self._fake(23.46, 1.12, 24.44, 1.17))
        self.assertFalse(report["significant"])
        self.assertIn("within the noise", format_comparison(report))

    def test_a_report_from_an_older_revision_still_renders(self):
        # stored comparisons are re-read later, so a report written before
        # `difference_sigmas` existed must derive it rather than raise
        old = {"ok": True, "corpus": "c.raw", "context": 512, "stride": 256, "keep": 4,
               "chunks": 195, "ppl_stock": 20.6123, "ppl_stream": 21.4478,
               "delta_ppl": 0.8355, "delta_pct": 4.05,
               "difference_stderr": 0.44, "significant": True,
               "stock": {"ppl": 20.6123, "ppl_stderr": 0.30447, "elapsed_s": 533.2},
               "stream": {"ppl": 21.4478, "ppl_stderr": 0.31760, "elapsed_s": 243.3,
                          "context_shifts": 131}}
        self.assertAlmostEqual(difference_sigmas(old), 1.90, places=2)
        text = format_comparison(old)
        self.assertIn("1.9 sigma", text)
        self.assertIn("larger than the noise", text)

    def test_failed_side_is_not_reported_as_a_gap(self):
        def run(binary, model, **kw):
            if kw.get("keep"):
                return {"ok": False, "error": "no PPL line in output", "exit_code": 1}
            return {"ok": True, "exit_code": 0, "elapsed_s": 1.0, "ppl": 23.4, "ppl_stderr": 0.3,
                    "chunks": kw.get("chunks"), "context": kw.get("context"), "corpus": "c.raw",
                    "stride": kw.get("stride"), "keep": kw.get("keep"),
                    "context_shift": kw.get("context_shift")}
        report = self._compare(run)
        self.assertFalse(report["ok"])
        self.assertNotIn("delta_ppl", report)
        self.assertIn("FAILED", format_comparison(report))


if __name__ == "__main__":
    unittest.main()
