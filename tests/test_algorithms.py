"""Tests for vipera.algorithms.

Each test asserts the *property* that makes the algorithm worth having — the
Guarantee the paper states, the exact identity that must hold, or the failure the
module must refuse — rather than that a function returned a number.
"""

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from vipera.algorithms import REGISTRY, get, list_algorithms, run
from vipera.algorithms import kv, quant, select, spec
from vipera.cli import main
from vipera.trace import save_trace, synth_trace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLMOE_TRACE = os.path.join(REPO_ROOT, "bench", "results", "olmoe-1b-7b-511tok.jsonl")


def _uniform(steps):
    """A lower-triangular stream of uniform attention rows."""
    return [[1.0 / (t + 1)] * (t + 1) for t in range(steps)]


class TestKVRetention(unittest.TestCase):
    def test_the_four_policies_are_offered(self):
        self.assertEqual(kv.POLICIES, ("recency", "sink-window", "h2o", "surprise"))

    def test_recency_retains_exactly_the_last_budget_positions(self):
        # Uniform rows of 1/(t+1), budget 2: the kept mass per step is
        # 1, 1, 2/3, 2/4 -> mean 0.791666...
        r = kv.simulate("recency", _uniform(4), budget=2)
        expected = (1.0 + 1.0 + 2 / 3 + 0.5) / 4
        self.assertAlmostEqual(r.retained_mass, expected, places=12)
        self.assertEqual(r.steps, 4)
        self.assertEqual(r.newest_would_drop, 0)

    def test_every_policy_drops_the_same_number_of_positions(self):
        # The budget is the same, so the eviction count is identical no matter
        # which positions each policy keeps. A difference here would mean one
        # policy silently ran at a different cache size.
        rows = kv.synth_attention(64, seed=2)
        results = kv.compare(rows, budget=16)
        counts = {r.evictions for r in results}
        self.assertEqual(len(counts), 1, f"policies evicted different counts: {counts}")
        self.assertEqual(counts.pop(), sum(max(0, (t + 1) - 16) for t in range(64)))

    def test_h2o_wins_the_objective_h2o_is_written_to_maximize(self):
        # Algorithm 1 maximizes the accumulated score, so it must win
        # `retained_accumulated`. Asserting this is what separates a faithful
        # implementation from one that merely keeps sensible tokens.
        rows = kv.synth_attention(128, seed=3)
        results = {r.policy: r for r in kv.compare(rows, budget=16)}
        best = max(results.values(), key=lambda r: r.retained_accumulated)
        self.assertEqual(best.policy, "h2o", [r.as_dict() for r in results.values()])

    def test_the_two_rankings_disagree(self):
        # The headline result: the policy that maximizes the accumulated score is
        # not the policy that maximizes what the next query needs, which is exactly
        # why the repo's measured `surprise` policy lost to `recency` on PPL.
        rows = kv.synth_attention(128, seed=3)
        by_mass = max(kv.compare(rows, budget=16), key=lambda r: r.retained_mass)
        by_acc = max(kv.compare(rows, budget=16), key=lambda r: r.retained_accumulated)
        self.assertNotEqual(by_mass.policy, by_acc.policy)

    def test_accumulated_policies_can_drop_the_newest_position(self):
        # llama.cpp needs KV position Y = X + 1, so an accumulated-score policy has
        # to be pinned. The count is reported rather than hidden: a policy that
        # needs the rule applied is not the policy in the paper.
        rows = kv.synth_attention(128, seed=3)
        results = {r.policy: r for r in kv.compare(rows, budget=16)}
        self.assertGreater(results["h2o"].newest_would_drop, 0)
        for p in ("recency", "sink-window", "surprise"):
            self.assertEqual(results[p].newest_would_drop, 0, p)

    def test_sink_window_pins_the_front_and_recency_does_not(self):
        rows = _uniform(64)
        # Budget 16 -> 4 sinks, so position 0 stays inside sink-window's budget.
        sink = kv.simulate("sink-window", rows, budget=16, sinks=4)
        rec = kv.simulate("recency", rows, budget=16, sinks=4)
        # On uniform attention both keep 16 of every live row, so only the *set*
        # differs; the accumulated share is what makes the sink visible.
        self.assertGreater(sink.retained_accumulated, rec.retained_accumulated)

    def test_empty_stream_is_zero_not_an_error(self):
        r = kv.simulate("h2o", [], budget=4)
        self.assertEqual(r.steps, 0)
        self.assertEqual(r.retained_mass, 0.0)
        self.assertEqual(r.evictions, 0)

    def test_budget_larger_than_the_stream_evicts_nothing(self):
        r = kv.simulate("recency", _uniform(3), budget=99)
        self.assertEqual(r.evictions, 0)
        self.assertAlmostEqual(r.retained_mass, 1.0, places=12)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            kv.simulate("turbo", _uniform(3), budget=2)
        with self.assertRaises(ValueError):
            kv.simulate("recency", _uniform(3), budget=0)
        with self.assertRaises(ValueError):
            kv.simulate("recency", [[1.0, 1.0], [1.0]], budget=2)
        with self.assertRaises(ValueError):
            kv.simulate("surprise", _uniform(3), budget=2, surprise=[1.0])

    def test_synth_attention_is_deterministic_and_well_formed(self):
        a = kv.synth_attention(32, seed=11)
        b = kv.synth_attention(32, seed=11)
        self.assertEqual(a, b)
        self.assertNotEqual(a, kv.synth_attention(32, seed=12))
        for t, row in enumerate(a):
            self.assertEqual(len(row), t + 1)
            self.assertGreater(row[t], 0.0)
            self.assertTrue(all(w >= 0.0 for w in row))

    def test_pure_streaming_regime_favours_a_window(self):
        # With no heavy hitters there is nothing for an accumulated-score rule to
        # find, so a window plus sinks should be the best policy on the per-step
        # metric. This is the regime StreamingLLM is documented for.
        rows = kv.synth_attention(128, seed=4, heavy_share=0.0, sink_share=0.3,
                                  recency_share=0.6)
        best = max(kv.compare(rows, budget=16), key=lambda r: r.retained_mass)
        self.assertEqual(best.policy, "sink-window")


class TestQuant(unittest.TestCase):
    def test_levels_match_the_symmetric_convention(self):
        self.assertEqual(quant.levels_for_bits(2), 1)
        self.assertEqual(quant.levels_for_bits(4), 7)
        self.assertEqual(quant.levels_for_bits(8), 127)
        with self.assertRaises(ValueError):
            quant.levels_for_bits(1)

    def test_rounding_is_symmetric_about_zero(self):
        # Python's round() is banker's rounding (round(0.5) == 0), which would bias
        # every quantizer. The helper must break the tie away from zero instead.
        self.assertEqual(quant._round_half_away(0.5), 1)
        self.assertEqual(quant._round_half_away(-0.5), -1)
        self.assertEqual(quant._round_half_away(1.5), 2)
        self.assertEqual(quant._round_half_away(-2.5), -3)

    def test_gptq_with_an_identity_hessian_is_exactly_round_to_nearest(self):
        # Exact identity, not a tolerance: with H = I the inverse is diagonal, the
        # off-diagonal coefficients are zero, and nothing is propagated.
        import random
        rng = random.Random(0)
        W = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(4)]
        n = 16
        H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        rtn = quant.quantize_rtn(W, bits=4)
        gptq = quant.quantize_gptq(W, H, bits=4)
        self.assertEqual(rtn, gptq)

    def test_gptq_with_a_diagonal_hessian_is_also_round_to_nearest(self):
        import random
        rng = random.Random(1)
        W = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(4)]
        H = [[0.0] * 16 for _ in range(16)]
        for i in range(16):
            H[i][i] = 0.5 + i / 10.0
        self.assertEqual(quant.quantize_rtn(W, bits=4), quant.quantize_gptq(W, H, bits=4))

    def test_a_group_of_one_cannot_propagate_anything(self):
        W = [[0.9, -0.7, 0.4, 0.13, -0.02], [0.31, 0.6, -0.55, 0.08, 0.44]]
        H = [[2.0 if i == j else 0.4 for j in range(5)] for i in range(5)]
        self.assertEqual(
            quant.quantize_rtn(W, bits=4, group=1),
            quant.quantize_gptq(W, H, bits=4, group=1),
        )

    def test_gptq_lowers_the_hessian_objective_on_a_real_hessian(self):
        # The paper's claim. A dense Hessian is what makes propagation possible; a
        # diagonal one provably cannot help, so this must use a dense one.
        import random
        rng = random.Random(7)
        n = 12
        A = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(48)]
        H = [[sum(A[r][i] * A[r][j] for r in range(48)) / 48 for j in range(n)]
             for i in range(n)]
        for i in range(n):
            H[i][i] += 0.05
        W = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(6)]
        rtn = quant.quantize_rtn(W, bits=4)
        gptq = quant.quantize_gptq(W, H, bits=4)
        self.assertLess(
            quant.hessian_objective(W, gptq, H),
            quant.hessian_objective(W, rtn, H),
        )

    def test_compare_reports_both_norms(self):
        # Reporting only the metric GPTQ optimizes is the defect docs/REVIEW.md was
        # written about, so the plain Frobenius number must always be present too —
        # and it is allowed to be worse, which is why no inequality is asserted on
        # it here.
        report = run("quant-gptq")
        for key in ("rtn_frobenius", "gptq_frobenius", "rtn_hessian", "gptq_hessian"):
            self.assertIn(key, report)
            self.assertGreater(report[key], 0.0)
        self.assertIn("hessian_improvement", report)

    def test_gptq_beats_round_to_nearest_on_the_registry_instance(self):
        # The win is instance-dependent: on a small block whose Hessian is close to
        # diagonal the propagated residual is smaller than the quantisation step and
        # the rounded values do not move at all. This pins the registry's own
        # instance, and the random-Hessian test below covers the general claim.
        report = run("quant-gptq")
        self.assertLess(report["gptq_hessian"], report["rtn_hessian"])
        self.assertGreater(report["hessian_improvement"], 1.0)

    def test_dense_inverse(self):
        ident = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(quant.dense_inverse(ident), ident)
        inv = quant.dense_inverse([[4.0, 7.0], [2.0, 6.0]])
        self.assertAlmostEqual(inv[0][0], 0.6, places=12)
        self.assertAlmostEqual(inv[0][1], -0.7, places=12)
        self.assertAlmostEqual(inv[1][0], -0.2, places=12)
        self.assertAlmostEqual(inv[1][1], 0.4, places=12)
        with self.assertRaises(ValueError):
            quant.dense_inverse([[1.0, 2.0], [2.0, 4.0]])   # singular
        with self.assertRaises(ValueError):
            quant.dense_inverse([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_zero_error_gives_zero_objectives(self):
        W = [[0.5, -0.25], [1.0, 0.0]]
        H = [[1.0, 0.2], [0.2, 1.0]]
        self.assertEqual(quant.frobenius_mse(W, W), 0.0)
        self.assertEqual(quant.hessian_objective(W, W, H), 0.0)

    def test_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            quant.quantize_rtn([], bits=4)
        with self.assertRaises(ValueError):
            quant.quantize_rtn([[1.0, 2.0], [3.0]], bits=4)
        with self.assertRaises(ValueError):
            quant.quantize_gptq([[1.0, 2.0]], [[1.0, 0.0]], bits=4)  # H is 2x2 vs 1x1
        with self.assertRaises(ValueError):
            quant.quantize_gptq([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]], bits=4, damp=-1)


class TestSelectSubmodular(unittest.TestCase):
    def _instance(self, seed, items=12, universe=20):
        state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        covers = []
        for _ in range(items):
            pick = []
            for _ in range(3):
                state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
                pick.append((state >> 33) % universe)
            covers.append(sorted(set(pick)))
        weights = [1.0 + 5.0 * ((e * 2654435761) % 31) / 31.0 for e in range(universe)]
        return select.coverage_objective(covers, weights)

    def test_coverage_is_monotone(self):
        f = self._instance(1)
        self.assertGreaterEqual(f(frozenset({0})), f(frozenset()))
        self.assertGreaterEqual(f(frozenset({0, 1, 2})), f(frozenset({0, 1})))

    def test_coverage_has_diminishing_returns(self):
        # Submodularity, checked directly on random nested sets: adding an element
        # to a bigger set can never help more than adding it to a smaller one.
        import random
        rng = random.Random(3)
        f = self._instance(2)
        ground = list(range(12))
        for _ in range(40):
            a_size = rng.randint(0, 3)
            b_size = rng.randint(a_size, 6)
            a = frozenset(rng.sample(ground, a_size))
            b = frozenset(set(a) | set(rng.sample([g for g in ground if g not in a],
                                                  b_size - a_size)))
            e = rng.choice([g for g in ground if g not in b])
            self.assertGreaterEqual(
                f(a | {e}) - f(a) + 1e-12,
                f(b | {e}) - f(b),
                "diminishing returns violated",
            )

    def test_lazy_greedy_returns_the_same_value_as_greedy(self):
        # CELF's speedup is only useful if it changes nothing that matters. The
        # guarantee is on the *value*, and that is what is asserted: when two
        # elements have exactly equal marginal gain the two algorithms may return
        # different sets of the same size and the same total, and pinning the set
        # would be pinning a tie-break rather than the algorithm.
        for seed in range(6):
            f = self._instance(seed)
            g = select.greedy(f, range(12), 4)
            lz = select.lazy_greedy(f, range(12), 4)
            self.assertAlmostEqual(lz.value, g.value, places=12, msg=f"seed {seed}")
            self.assertEqual(len(lz.chosen), len(g.chosen), f"seed {seed}")
            self.assertEqual(len(set(lz.chosen)), len(lz.chosen), "no duplicates")

    def test_lazy_greedy_uses_fewer_evaluations(self):
        f = self._instance(4)
        g = select.greedy(f, range(12), 4)
        lz = select.lazy_greedy(f, range(12), 4)
        self.assertLess(lz.evaluations, g.evaluations)

    def test_greedy_meets_the_one_minus_one_over_e_bound(self):
        bound = 1.0 - 1.0 / math.e
        for seed in range(5):
            f = self._instance(seed)
            g = select.greedy(f, range(12), 3)
            _, opt = select.brute_force(f, range(12), 3)
            self.assertGreaterEqual(g.value + 1e-12, bound * opt, f"seed {seed}")

    def test_brute_force_finds_the_true_optimum(self):
        # Check the checker against an independent exhaustive walk.
        import itertools
        f = self._instance(9)
        chosen, value = select.brute_force(f, range(8), 3)
        best = max(f(frozenset(c)) for c in itertools.combinations(range(8), 3))
        self.assertAlmostEqual(value, best, places=12)
        self.assertEqual(len(chosen), 3)

    def test_k_zero_selects_nothing(self):
        f = self._instance(5)
        self.assertEqual(select.greedy(f, range(12), 0).chosen, [])
        self.assertEqual(select.lazy_greedy(f, range(12), 0).chosen, [])
        self.assertEqual(select.brute_force(f, range(12), 0)[0], [])

    def test_k_larger_than_the_ground_set_selects_everything(self):
        f = self._instance(6)
        self.assertEqual(len(select.greedy(f, range(6), 99).chosen), 6)

    def test_a_saturated_objective_stops_early(self):
        # A zero-gain step must not spend the rest of the budget: the result is
        # identical either way, and stopping is what keeps CELF's evaluation count
        # honest.
        f = lambda _s: 0.0
        self.assertEqual(select.greedy(f, range(10), 5).chosen, [])
        self.assertEqual(select.lazy_greedy(f, range(10), 5).chosen, [])

    def test_rejects_empty_weights(self):
        with self.assertRaises(ValueError):
            select.coverage_objective([[1, 2]], [])

    def test_negative_k_is_refused(self):
        with self.assertRaises(ValueError):
            select.greedy(lambda _s: 0.0, range(3), -1)
        with self.assertRaises(ValueError):
            select.lazy_greedy(lambda _s: 0.0, range(3), -1)


class TestSpeculative(unittest.TestCase):
    def test_acceptance_rule(self):
        self.assertEqual(spec.accept_prob(0.5, 0.5), 1.0)
        self.assertAlmostEqual(spec.accept_prob(0.1, 0.5), 0.2, places=12)
        self.assertEqual(spec.accept_prob(0.5, 0.0), 0.0)
        self.assertEqual(spec.accept_prob(0.9, 0.1), 1.0)

    def test_overlap_with_itself_is_one(self):
        p = [0.2, 0.3, 0.5]
        self.assertAlmostEqual(spec.overlap(p, p), 1.0, places=12)

    def test_an_identical_draft_accepts_every_position(self):
        p = [0.1, 0.6, 0.3]
        self.assertAlmostEqual(spec.expected_accepted([p] * 5, [p] * 5), 5.0, places=9)

    def test_no_drafted_positions_accepts_nothing(self):
        self.assertEqual(spec.expected_accepted([], []), 0.0)

    def test_expected_accepted_never_exceeds_the_draft_length(self):
        p = [0.25, 0.25, 0.25, 0.25]
        q = [0.4, 0.3, 0.2, 0.1]
        self.assertLessEqual(spec.expected_accepted([p] * 6, [q] * 6), 6.0 + 1e-9)

    def test_closed_form_agrees_with_monte_carlo(self):
        # The whole reason the closed form is trustworthy: it is checked against
        # the sampler it claims to describe.
        rows = [
            ([0.6, 0.3, 0.1], [0.45, 0.3, 0.25]),
            ([0.5, 0.5], [0.5, 0.5]),
            ([0.34, 0.33, 0.33], [0.2, 0.5, 0.3]),
            ([0.7, 0.2, 0.1], [0.4, 0.35, 0.25]),
        ]
        t = [p for p, _ in rows]
        d = [q for _, q in rows]
        self.assertAlmostEqual(
            spec.expected_accepted(t, d), spec.measure_accepted(t, d, trials=40000, seed=5),
            delta=0.05,
        )

    def test_residual_sums_to_one_and_is_refused_when_empty(self):
        p = [0.6, 0.3, 0.1]
        q = [0.4, 0.3, 0.3]
        self.assertAlmostEqual(math.fsum(spec.residual(p, q)), 1.0, places=12)
        with self.assertRaises(ValueError):
            spec.residual(p, p)

    def test_rejection_sampling_preserves_the_target_distribution(self):
        # The exactness claim, tested as a distribution rather than a formula: a
        # wrong residual or a wrong accept rule shows up as a shifted histogram.
        p = [0.5, 0.3, 0.15, 0.05]
        q = [0.1, 0.2, 0.3, 0.4]
        rng = spec._Rng(11)
        counts = [0] * len(p)
        trials = 60000
        for _ in range(trials):
            x = rng.choice(q)
            if rng.next_float() < spec.accept_prob(p[x], q[x]):
                counts[x] += 1
            else:
                counts[rng.choice(spec.residual(p, q))] += 1
        for i, expected in enumerate(p):
            self.assertAlmostEqual(counts[i] / trials, expected, delta=0.02,
                                   msg=f"value {i}")

    def test_tokens_per_step_is_at_least_one(self):
        p = [0.9, 0.1]
        q = [0.1, 0.9]
        self.assertGreaterEqual(spec.expected_tokens_per_step([p] * 4, [q] * 4), 1.0)

    def test_sweep_is_non_decreasing_and_reports_its_best_k(self):
        def position(i):
            p = [0.6, 0.3, 0.1]
            g = 0.2 + 0.05 * i
            q = [x * (1 - g) + g / 3 for x in p]
            s = math.fsum(q)
            return p, [x / s for x in q]

        sweep = spec.sweep_draft_length(position, max_k=6, cost_ratio=0.0)
        self.assertEqual([r["k"] for r in sweep.rows], list(range(7)))
        acc = [r["expected_accepted"] for r in sweep.rows]
        self.assertEqual(acc, sorted(acc))
        self.assertEqual(sweep.best_k(), 6)  # free drafting -> more is always better

    def test_cost_ratio_shifts_the_optimum_down(self):
        def position(i):
            p = [0.6, 0.3, 0.1]
            g = 0.2 + 0.05 * i
            q = [x * (1 - g) + g / 3 for x in p]
            s = math.fsum(q)
            return p, [x / s for x in q]

        free = spec.sweep_draft_length(position, max_k=12, cost_ratio=0.0).best_k()
        priced = spec.sweep_draft_length(position, max_k=12, cost_ratio=1.0).best_k()
        self.assertLess(priced, free)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            spec.overlap([0.5, 0.5], [0.4, 0.5, 0.1])
        with self.assertRaises(ValueError):
            spec.overlap([0.5, 0.4], [0.5, 0.5])
        with self.assertRaises(ValueError):
            spec.expected_accepted([[0.5, 0.5]], [])
        with self.assertRaises(ValueError):
            spec.measure_accepted([[0.5, 0.5]], [[0.5, 0.5]], trials=0)
        with self.assertRaises(ValueError):
            spec.speedup([[0.5, 0.5]], [[0.5, 0.5]], cost_ratio=-1)


class TestRegistry(unittest.TestCase):
    def test_registry_lists_four_algorithms_with_unique_names(self):
        names = [a.name for a in list_algorithms()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names),
                         {"kv-retention", "quant-gptq", "select-submodular",
                          "spec-draft-length"})
        self.assertEqual({a.kind for a in list_algorithms()}, {"kv", "quant", "select", "spec"})

    def test_every_entry_names_a_source_and_a_status(self):
        for a in REGISTRY:
            self.assertTrue(a.source.strip(), a.name)
            self.assertIn(a.status, {"exact", "simulated", "measured"}, a.name)

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            get("nope")

    def test_every_algorithm_returns_json_serialisable_output(self):
        for a in list_algorithms():
            report = run(a.name)
            self.assertIsInstance(report, dict, a.name)
            json.dumps(report)  # must not raise


class TestAlgorithmCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_algo_list_prints_every_name(self):
        code, out, _ = self._run(["algo", "list"])
        self.assertEqual(code, 0)
        for name in ("kv-retention", "quant-gptq", "select-submodular", "spec-draft-length"):
            self.assertIn(name, out)

    def test_algo_list_json_parses(self):
        code, out, _ = self._run(["algo", "list", "--json"])
        self.assertEqual(code, 0)
        rows = json.loads(out)
        self.assertEqual(len(rows), 4)
        self.assertIn("source", rows[0])

    def test_algo_run_reports_both_quantisation_norms(self):
        code, out, _ = self._run(["algo", "run", "quant-gptq", "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertIn("rtn_hessian", report)
        self.assertIn("gptq_hessian", report)

    def test_algo_run_on_the_real_trace_reports_a_modular_objective(self):
        if not os.path.exists(OLMOE_TRACE):
            self.skipTest(f"no captured trace at {OLMOE_TRACE}")
        code, out, _ = self._run(["algo", "run", "select-submodular",
                                  "--trace", OLMOE_TRACE, "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)["trace_mode"]
        # The trace objective is modular, so greedy must tie the optimum exactly:
        # that is the finding, not a tolerance.
        self.assertFalse(report["submodular_bound_applies"])
        self.assertEqual(report["min_ratio_to_optimum"], 1.0)

    def test_the_human_report_names_every_key_it_was_given(self):
        """Regression: the flat printer silently dropped every nested value.

        `select-submodular` returns only dicts at the top level, so the human path
        printed the header and nothing else, and `spec-draft-length` lost three of
        its five keys. The printed numbers are supposed to be the ones the algorithm
        produced, so a key that cannot be rendered has to be a visible failure
        instead of a shorter report.
        """
        from vipera.algorithms import list_algorithms, run as run_algorithm

        for algorithm in list_algorithms():
            with self.subTest(algorithm=algorithm.name):
                code, out, _ = self._run(["algo", "run", algorithm.name])
                self.assertEqual(code, 0)
                for key in run_algorithm(algorithm.name):
                    self.assertIn(key, out, f"{algorithm.name}: {key!r} was dropped")

    def test_the_human_report_renders_a_report_made_only_of_dicts(self):
        # The trace-mode result is a dict of dicts, which is exactly the shape the
        # old printer reduced to a bare header.
        if not os.path.exists(OLMOE_TRACE):
            self.skipTest(f"no captured trace at {OLMOE_TRACE}")
        code, out, _ = self._run(["algo", "run", "select-submodular",
                                  "--trace", OLMOE_TRACE])
        self.assertEqual(code, 0)
        for key in ("trace_mode", "per_layer", "ratio_to_optimum", "synthetic_instance"):
            self.assertIn(key, out, f"{key!r} was dropped from the human report")

    def test_unknown_algorithm_is_exit_2(self):
        code, _, err = self._run(["algo", "run", "does-not-exist"])
        self.assertEqual(code, 2)
        self.assertIn("unknown algorithm", err)

    def test_a_trace_on_an_algorithm_that_ignores_it_is_exit_2(self):
        # Silently accepting it would let the caller believe the numbers were
        # trace-driven.
        code, _, err = self._run(["algo", "run", "quant-gptq", "--trace", "x.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("does not consume a trace", err)

    def test_trace_mode_refuses_an_empty_trace(self):
        path = os.path.join(self.tmp.name, "empty.jsonl")
        with open(path, "w") as f:
            f.write("")
        code, _, err = self._run(["algo", "run", "select-submodular", "--trace", path])
        self.assertEqual(code, 2)
        self.assertTrue(err.strip())


if __name__ == "__main__":
    unittest.main()
