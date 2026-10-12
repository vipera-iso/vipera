import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from tests._fixture import make_gguf
from vipera.cli import main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")
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
