import json
import os
import tempfile
import unittest

from bench.capture_trace import build_command, summarise


class TestBuildCommand(unittest.TestCase):
    def test_includes_model_prompt_and_token_budget(self):
        cmd = build_command("llama-cli", "m.gguf", prompt="hi", tokens=32,
                            threads=4, ngl=0, seed=1, extra=[])
        self.assertEqual(cmd[0], "llama-cli")
        self.assertIn("-m", cmd)
        self.assertEqual(cmd[cmd.index("-m") + 1], "m.gguf")
        self.assertEqual(cmd[cmd.index("-n") + 1], "32")
        self.assertEqual(cmd[cmd.index("-t") + 1], "4")
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "0")

    def test_extra_flags_are_passed_through_unchanged(self):
        cmd = build_command("llama-cli", "m.gguf", prompt="hi", tokens=1,
                            threads=None, ngl=None, seed=0, extra=["--top-k", "40"])
        self.assertEqual(cmd[-2:], ["--top-k", "40"])


class TestSummarise(unittest.TestCase):
    def test_reports_shape_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"t": 0, "layers": [[0, 1], [2, 3]]}) + "\n")
                f.write(json.dumps({"t": 1, "layers": [[0, 1], [2, 4]]}) + "\n")
            info = summarise(path)
        self.assertEqual(info["tokens"], 2)
        self.assertEqual(info["layers"], 2)
        self.assertEqual(info["top_k"], 2)
        # layer 0 only ever sees {0,1}; layer 1 sees {2,3,4}.
        self.assertEqual(info["distinct_experts_per_layer"], [2, 3])


if __name__ == "__main__":
    unittest.main()
