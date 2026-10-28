import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from tests._fixture import (make_dense_gguf, make_gguf, make_gguf_with_expert_data,
                            make_unknown_quant_gguf)
from vipera.cli import main
from vipera.gguf import read_gguf
from vipera.schedule import Quant, plan_residency
from vipera.arch import Architecture
from vipera.manifest import Target
from vipera.trace import save_trace, synth_trace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")
GLM = os.path.join(REPO_ROOT, "arch", "glm-5.3-flash.toml")
OLMOE = os.path.join(REPO_ROOT, "arch", "olmoe-1b-7b.toml")
LLAMA = os.path.join(REPO_ROOT, "arch", "llama3-70b.toml")
LLAMA16 = os.path.join(REPO_ROOT, "hw", "laptop16-igpu.toml")


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.gguf")
        make_gguf(self.path, block_count=2, n_experts=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyze_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["analyze", self.path, "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["n_experts"], 4)
        self.assertEqual(data["n_layers"], 2)
        self.assertIn("always_resident", data["bytes"])

    def test_plan_writes_manifest(self):
        outdir = os.path.join(self.tmp.name, "out")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["plan", self.path, "--target", HW, "--out", outdir])
        self.assertEqual(code, 0)
        written = os.path.join(outdir, "tiny.vipera.json")
        self.assertTrue(os.path.exists(written))
        with open(written) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["vipera_version"], 1)
        self.assertEqual(manifest["target"]["name"], "gtx1650-laptop")

    def test_plan_hardware_profile_is_real(self):
        self.assertTrue(os.path.exists(HW), f"missing hardware profile at {HW}")

    def test_trace_target_derives_the_matched_cache_size(self):
        path = os.path.join(self.tmp.name, "trace.jsonl")
        save_trace(path, synth_trace(12, n_layers=45, n_experts=288, top_k=8, seed=0))
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["trace", path, "--arch", GLM, "--target", HW,
                         "--quant", "q4-experts", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        expected = plan_residency(
            Architecture.from_toml(GLM), Target.from_toml(HW), quant=Quant.q4_experts()
        )
        self.assertEqual(data["slots"], expected.slots)
        self.assertTrue(data["cache_size_source"].startswith("matched"))
        self.assertIn("global_static_cache_hit_ceiling", data)

    def test_schedule_trace_consumes_the_whole_trace(self):
        path = os.path.join(self.tmp.name, "trace.jsonl")
        save_trace(path, synth_trace(30, n_layers=45, n_experts=288, top_k=8, seed=0))
        outdir = os.path.join(self.tmp.name, "out")
        with redirect_stdout(io.StringIO()):
            code = main(["schedule", GLM, "--target", HW, "--quant", "q4-experts",
                         "--trace", path, "--cache-policy", "hot", "--out", outdir])
        self.assertEqual(code, 0)
        with open(os.path.join(outdir, "glm-5.3-flash.vipera.json")) as f:
            manifest = json.load(f)
        # All 30 tokens, not the modelled default of 8 (which round 1 silently used).
        self.assertEqual(manifest["expert_cache"]["eval_window"]["end"], 30)
        self.assertEqual(manifest["expert_cache"]["policy"], "hot")
        self.assertIn("ceiling_ratio", manifest["expert_cache"])

    def test_prune_gated_json(self):
        path = os.path.join(self.tmp.name, "trace.jsonl")
        save_trace(path, synth_trace(20, n_layers=45, n_experts=288, top_k=8, seed=0))
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["prune-gated", path, "--arch", GLM,
                         "--max-keep-fraction", "0.5", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["algorithm"], "gated-expert-pruning")
        self.assertEqual(data["rolled_back_layers"] + data["pruned_layers"], 45)

    def test_sweep_runs_on_a_small_model(self):
        # Regression: sweep asks for keep_layers=25, which used to crash on a
        # model with fewer than 25 layers.
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["sweep", OLMOE, "--target", HW])
        self.assertEqual(code, 0)
        self.assertIn("all values MODELLED", out.getvalue())

    def test_rewrite_prunes_experts_from_a_trace(self):
        # the write side of `prune`: the plan becomes a file whose expert
        # dimension actually shrank, which is what every format paper needed.
        source = os.path.join(self.tmp.name, "source.gguf")
        make_gguf_with_expert_data(source, block_count=2, n_experts=4)
        trace = os.path.join(self.tmp.name, "rtrace.jsonl")
        save_trace(trace, synth_trace(5, n_layers=2, n_experts=4, top_k=2, seed=0))
        dest = os.path.join(self.tmp.name, "pruned.gguf")

        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["rewrite", source, "--out", dest, "--from-trace", trace,
                         "--prune-to", "2", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["experts_before"], 8)
        self.assertEqual(data["experts_after"], 4)
        self.assertLess(data["bytes_after"], data["bytes_before"])
        before = read_gguf(source).tensor("blk.1.ffn_gate_exps.weight").shape
        after = read_gguf(dest).tensor("blk.1.ffn_gate_exps.weight").shape
        self.assertEqual(after, before[:2] + (2,), "expert dimension must shrink to 2")

    def test_rewrite_needs_a_source_of_expert_order(self):
        source = os.path.join(self.tmp.name, "source.gguf")
        make_gguf_with_expert_data(source, block_count=2, n_experts=4)
        err = io.StringIO()
        with redirect_stderr(err):
            code = main(["rewrite", source, "--out", os.path.join(self.tmp.name, "o.gguf")])
        self.assertEqual(code, 2)
        self.assertIn("--from-trace or --random", err.getvalue())

    def test_rewrite_random_reorders_without_changing_the_size(self):
        source = os.path.join(self.tmp.name, "source.gguf")
        make_gguf_with_expert_data(source, block_count=2, n_experts=4)
        dest = os.path.join(self.tmp.name, "shuffled.gguf")
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["rewrite", source, "--out", dest, "--random", "7", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["experts_before"], data["experts_after"])
        self.assertEqual(data["bytes_before"], data["bytes_after"])

    def test_rewrite_refuses_a_model_without_routed_experts(self):
        dense = os.path.join(self.tmp.name, "dense.gguf")
        make_dense_gguf(dense, block_count=2)
        err = io.StringIO()
        with redirect_stderr(err):
            code = main(["rewrite", dense, "--out", os.path.join(self.tmp.name, "o.gguf"),
                         "--random", "1"])
        self.assertEqual(code, 2)
        self.assertIn("no routed-expert tensor", err.getvalue())

    def test_rewrite_refuses_a_trace_covering_layers_without_experts(self):
        source = os.path.join(self.tmp.name, "source.gguf")
        make_gguf_with_expert_data(source, block_count=2, n_experts=4)
        trace = os.path.join(self.tmp.name, "wide.jsonl")
        save_trace(trace, synth_trace(4, n_layers=4, n_experts=4, top_k=2, seed=0))
        err = io.StringIO()
        with redirect_stderr(err):
            code = main(["rewrite", source, "--out", os.path.join(self.tmp.name, "o.gguf"),
                         "--from-trace", trace])
        self.assertEqual(code, 2)
        self.assertIn("no routed experts", err.getvalue())

    def test_score_fails_the_local_llm_16gb_instance(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["score", LLAMA, "--target", LLAMA16,
                         "--quality", "99", "--power", "65"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("FAIL", text)
        self.assertIn("tokens/s", text)


class TestCLIErrorExitCodes(unittest.TestCase):
    """Bad input must exit 2, not 1 (which `plan` uses to mean "doesn't fit")."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.gguf")
        make_gguf(self.path, block_count=2, n_experts=4)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(argv)

    def test_missing_model_file_is_exit_2(self):
        code = self._run(["analyze", os.path.join(self.tmp.name, "nope.gguf")])
        self.assertEqual(code, 2)

    def test_unreadable_target_is_exit_2(self):
        code = self._run(["plan", self.path, "--target", os.path.join(self.tmp.name, "nope.toml")])
        self.assertEqual(code, 2)

    def test_bad_toml_is_exit_2(self):
        bad = os.path.join(self.tmp.name, "bad.toml")
        with open(bad, "w") as f:
            f.write("this is = = not toml\n")
        code = self._run(["plan", self.path, "--target", bad])
        self.assertEqual(code, 2)

    def test_target_missing_a_key_is_exit_2(self):
        incomplete = os.path.join(self.tmp.name, "incomplete.toml")
        with open(incomplete, "w") as f:
            f.write('name = "x"\n')  # no vram_mb / ram_mb
        code = self._run(["plan", self.path, "--target", incomplete])
        self.assertEqual(code, 2)

    def test_unknown_quant_model_is_exit_2(self):
        path = os.path.join(self.tmp.name, "mxfp4.gguf")
        make_unknown_quant_gguf(path, ggml_type=999)
        code = self._run(["analyze", path])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
