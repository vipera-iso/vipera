import io
import json
import os
import stat
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout

from bench.bench import (
    build_command,
    main,
    parse_llama_bench_json,
    run_bench,
    split_args,
    test_label,
    total_tokens,
)
from bench.measure import bytes_per_token, read_proc_io, read_rss_kb

# llama-bench's JSON has no `test` field (verified against llama.cpp f2cc728), so the
# fixture deliberately omits it — a fixture with the field would hide the bug it caused.
RESULTS = [
    {"n_prompt": 512, "n_gen": 0, "avg_ts": 123.4, "stddev_ts": 1.0, "model_filename": "stub.gguf"},
    {"n_prompt": 0, "n_gen": 128, "avg_ts": 12.3, "stddev_ts": 0.5, "model_filename": "stub.gguf"},
]
GOOD_JSON = json.dumps(RESULTS)

# Stubs are Python scripts, not shell one-liners: embedding a JSON string inside
# `sh -c "…"` mangles the quotes and makes the test lie about what it verifies.
STUB_OK = """\
#!/usr/bin/env python3
import json, sys, time
buf = bytearray(16 * 1024 * 1024)   # make RSS measurable
time.sleep(0.4)
json.dump(%s, sys.stdout)
""" % (RESULTS,)

STUB_FAIL = """\
#!/usr/bin/env python3
import sys
print("boom: model not found", file=sys.stderr)
sys.exit(3)
"""

STUB_GARBAGE = """\
#!/usr/bin/env python3
print("this is not json")
"""

# Records the argv it was handed, so the test can see what reached llama-bench.
STUB_ARGV = """\
#!/usr/bin/env python3
import json, os, sys
with open(os.path.abspath(__file__) + ".argv", "w") as f:
    json.dump(sys.argv, f)
print("[]")
"""


def _write_stub(path: str, script: str) -> str:
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


class TestHelpers(unittest.TestCase):
    def test_build_command(self):
        cmd = build_command("llama-bench", "m.gguf", ngl=0, threads=8, pp=64, tg=16,
                            extra=["--n-cpu-moe", "2"])
        self.assertEqual(cmd[0], "llama-bench")
        self.assertIn("json", cmd)
        self.assertIn("-ngl", cmd)
        self.assertEqual(cmd[-2:], ["--n-cpu-moe", "2"])

    def test_parse_accepts_both_shapes(self):
        self.assertEqual(len(parse_llama_bench_json(GOOD_JSON)), 2)
        self.assertEqual(len(parse_llama_bench_json(json.dumps({"results": RESULTS}))), 2)

    def test_parse_rejects_other_json(self):
        with self.assertRaises(ValueError):
            parse_llama_bench_json(json.dumps({"nothing": 1}))

    def test_total_tokens(self):
        self.assertEqual(total_tokens(parse_llama_bench_json(GOOD_JSON)), 640)

    def test_label_is_derived_when_absent(self):
        self.assertEqual(test_label({"n_prompt": 512, "n_gen": 0}), "pp512")
        self.assertEqual(test_label({"n_prompt": 0, "n_gen": 128}), "tg128")
        self.assertEqual(test_label({"n_prompt": 256, "n_gen": 64}), "pp256+tg64")
        self.assertEqual(test_label({}), "unknown")

    def test_label_is_kept_when_present(self):
        self.assertEqual(test_label({"test": "pp512", "n_prompt": 512}), "pp512")

    def test_bytes_per_token_guards(self):
        self.assertIsNone(bytes_per_token(None, 100))
        self.assertIsNone(bytes_per_token(1000, 0))
        self.assertEqual(bytes_per_token(1000, 100), 10.0)

    def test_proc_helpers_on_self(self):
        self.assertIsNotNone(read_rss_kb(os.getpid()))
        self.assertIsNotNone(read_proc_io(os.getpid()))


class TestRunBench(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = os.path.join(self.tmp.name, "stub-bench")

    def tearDown(self):
        self.tmp.cleanup()

    def test_successful_run_collects_metrics(self):
        _write_stub(self.bin, STUB_OK)
        report = run_bench(self.bin, "stub.gguf", gpu=False, sample_interval=0.05)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["results"][1]["avg_ts"], 12.3)
        self.assertEqual(report["results"][0]["test"], "pp512")
        self.assertEqual(report["results"][1]["test"], "tg128")
        self.assertEqual(report["tokens_processed"], 640)
        self.assertGreater(report["peak_rss_kb"], 0)
        self.assertGreater(report["elapsed_s"], 0.0)

    def test_nonzero_exit_is_not_a_pass(self):
        _write_stub(self.bin, STUB_FAIL)
        report = run_bench(self.bin, "stub.gguf", gpu=False, sample_interval=0.05)
        self.assertFalse(report["ok"])
        self.assertEqual(report["exit_code"], 3)
        self.assertIn("boom", report["stderr_tail"])

    def test_unparseable_output_is_not_a_pass(self):
        _write_stub(self.bin, STUB_GARBAGE)
        report = run_bench(self.bin, "stub.gguf", gpu=False, sample_interval=0.05)
        self.assertFalse(report["ok"])
        self.assertTrue(report["error"].startswith("could not parse"))

    def test_main_returns_2_on_unparseable(self):
        _write_stub(self.bin, STUB_GARBAGE)
        with redirect_stdout(io.StringIO()):
            code = main(["--bin", self.bin, "--model", "stub.gguf", "--no-gpu", "--interval", "0.05"])
        self.assertEqual(code, 2)

    def test_main_writes_json_report(self):
        _write_stub(self.bin, STUB_OK)
        path = os.path.join(self.tmp.name, "out", "r.json")
        with redirect_stdout(io.StringIO()):
            code = main(["--bin", self.bin, "--model", "stub.gguf", "--no-gpu",
                         "--interval", "0.05", "--json", path])
        self.assertEqual(code, 0)
        with open(path) as f:
            saved = json.load(f)
        self.assertEqual(saved["tokens_processed"], 640)


class TestPassthroughFlags(unittest.TestCase):
    """llama-bench flags bench does not define must reach llama-bench unchanged."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = os.path.join(self.tmp.name, "stub-bench")

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_args_keeps_unknown_flags_in_order(self):
        _, extra = split_args(["--model", "m", "-r", "1", "--no-warmup"])
        self.assertEqual(extra, ["-r", "1", "--no-warmup"])

    def test_split_args_drops_the_separator(self):
        _, extra = split_args(["--model", "m", "--", "--n-cpu-moe", "2"])
        self.assertEqual(extra, ["--n-cpu-moe", "2"])

    def test_documented_flags_reach_llama_bench_without_a_separator(self):
        _write_stub(self.bin, STUB_ARGV)
        with redirect_stdout(io.StringIO()):
            code = main(["--bin", self.bin, "--model", "stub.gguf", "--no-gpu",
                         "--interval", "0.05", "-r", "1", "--no-warmup"])
        self.assertEqual(code, 0)
        with open(self.bin + ".argv") as f:
            argv = json.load(f)
        self.assertEqual(argv[-3:], ["-r", "1", "--no-warmup"])


if __name__ == "__main__":
    unittest.main()
