import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from tests._fixture import make_gguf
from vipera.cli import main
from vipera.schedule import Quant, plan_residency
from vipera.arch import Architecture
from vipera.manifest import Target
from vipera.trace import save_trace, synth_trace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")
GLM = os.path.join(REPO_ROOT, "arch", "glm-5.3-flash.toml")
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

    def test_score_fails_the_local_llm_16gb_instance(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["score", LLAMA, "--target", LLAMA16,
                         "--quality", "99", "--power", "65"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("FAIL", text)
        self.assertIn("tokens/s", text)


if __name__ == "__main__":
    unittest.main()
