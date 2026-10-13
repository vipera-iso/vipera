import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from tests._fixture import make_gguf
from vipera.cli import main
from vipera.run import (
    RunFlags,
    build_command,
    detect_gpu_backend,
    format_compare,
    parse_prefetch_stats,
    parse_throughput,
    plan_flags,
    supports_prefetch,
)
from vipera.trace import save_trace, synth_trace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")

MANIFEST = {
    "vipera_version": 1,
    "source": {"architecture": "deepseek2", "n_layers": 27, "n_experts": 64},
    "placement": {
        "resident_vram_bytes": 833058816,
        "resident_ram_bytes": 0,
        "expert_cache_vram_bytes": 2925037568,
        "expert_ram_bytes": 6602323968,
        "expert_spill_disk_bytes": 0,
    },
    "kv_cache": {"type_k": "q8_0", "type_v": "q8_0"},
}


class TestPlanFlags(unittest.TestCase):
    def test_gpu_target_maps_experts_to_cpu_and_cache_to_vram(self):
        flags = plan_flags(MANIFEST, gpu=True)
        self.assertEqual(flags.ngl, 27)
        self.assertEqual(flags.n_cpu_moe, 27)
        # 2925037568 B / MiB
        self.assertEqual(flags.moe_cache_mib, 2925037568 // (1024 * 1024))
        self.assertEqual(flags.cache_type_k, "q8_0")
        self.assertEqual(flags.cache_type_v, "q8_0")
        self.assertEqual(flags.warnings, [])

    def test_cpu_only_build_warns_instead_of_pretending(self):
        flags = plan_flags(MANIFEST, gpu=False)
        self.assertEqual(flags.ngl, 0)
        self.assertIsNone(flags.n_cpu_moe)
        self.assertIsNone(flags.moe_cache_mib)
        self.assertEqual(len(flags.warnings), 1)
        self.assertIn("without a GPU backend", flags.warnings[0])

    def test_disk_spill_is_reported_as_unmappable(self):
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["placement"]["expert_spill_disk_bytes"] = 1024 * 1024 * 500
        flags = plan_flags(manifest, gpu=True)
        self.assertTrue(any("no per-expert disk streaming" in w for w in flags.warnings))

    def test_live_layer_count_overrides_the_manifest(self):
        flags = plan_flags(MANIFEST, gpu=True, n_layers=99)
        self.assertEqual(flags.ngl, 99)
        self.assertEqual(flags.n_cpu_moe, 99)


class TestBuildCommand(unittest.TestCase):
    def test_flags_reach_llama_cli(self):
        flags = plan_flags(MANIFEST, gpu=True)
        cmd = build_command(
            "/bin/llama-cli", "model.gguf", flags,
            context=512, n_predict=8, prompt="hi", threads=4,
        )
        self.assertEqual(cmd[0], "/bin/llama-cli")
        for pair in (("-m", "model.gguf"), ("-c", "512"), ("-n", "8"), ("-p", "hi"),
                     ("-ngl", "27"), ("--n-cpu-moe", "27"), ("-ctk", "q8_0"),
                     ("-ctv", "q8_0"), ("-t", "4")):
            i = cmd.index(pair[0])
            self.assertEqual(cmd[i + 1], pair[1], f"{pair[0]} value")
        # single turn so the process exits instead of waiting on a TTY
        self.assertIn("-st", cmd)

    def test_extra_flags_are_appended_last(self):
        cmd = build_command(
            "/bin/llama-cli", "m.gguf", RunFlags(ngl=0),
            context=256, n_predict=4, prompt="x", extra=["--no-warmup"],
        )
        self.assertEqual(cmd[-1], "--no-warmup")

    def test_prefetch_turns_on_verbose_logging(self):
        # The MoE cache and the prefetch verdict only log at INFO level.
        base = build_command(
            "/bin/llama-cli", "m.gguf", RunFlags(),
            context=256, n_predict=4, prompt="x",
        )
        self.assertNotIn("-v", base)
        with_pf = build_command(
            "/bin/llama-cli", "m.gguf", RunFlags(),
            context=256, n_predict=4, prompt="x", prefetch=True,
        )
        self.assertIn("-v", with_pf)
        # an explicit verbosity from the user is not duplicated
        explicit = build_command(
            "/bin/llama-cli", "m.gguf", RunFlags(),
            context=256, n_predict=4, prompt="x", prefetch=True, extra=["--verbose"],
        )
        self.assertEqual(explicit.count("-v"), 0)
        self.assertEqual(explicit.count("--verbose"), 1)


class TestParseThroughput(unittest.TestCase):
    def test_last_line_wins(self):
        text = "[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]\n" \
               "[ Prompt: 12.2 t/s | Generation: 13.7 t/s ]\n"
        self.assertEqual(parse_throughput(text), (12.2, 13.7))

    def test_missing_line_is_none(self):
        self.assertEqual(parse_throughput("nothing here"), (None, None))

    def test_parses_the_patches_prefetch_report(self):
        line = ("llama_moe_cache: prefetch: placed = 541, useful = 300 "
                "(55.45%), extra upload = 120.50 MiB\n")
        self.assertEqual(
            parse_prefetch_stats(line),
            {"placed": 541, "useful": 300, "useful_pct": 55.45, "extra_upload_mib": 120.5},
        )

    def test_absent_prefetch_report_is_none(self):
        self.assertIsNone(parse_prefetch_stats("no prefetch here"))


class TestDetectGpu(unittest.TestCase):
    def test_backend_library_next_to_binary(self):
        with tempfile.TemporaryDirectory() as d:
            binary = os.path.join(d, "llama-cli")
            open(binary, "w").close()
            self.assertFalse(detect_gpu_backend(binary))
            open(os.path.join(d, "libggml-cuda.so"), "w").close()
            self.assertTrue(detect_gpu_backend(binary))


def _fake_llama_cli(directory: str) -> str:
    """A stand-in binary that prints llama-cli's throughput line."""
    path = os.path.join(directory, "llama-cli")
    with open(path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write('echo "[ Prompt: 10.0 t/s | Generation: 7.5 t/s ]"\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


class TestRunCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.gguf")
        make_gguf(self.path, block_count=2, n_experts=4)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_prints_the_mapped_command(self):
        # A GPU-enabled build so the mapping is the VRAM-with-CPU-experts path
        # rather than the CPU-only fallback.
        binary = os.path.join(self.tmp.name, "llama-cli")
        open(binary, "w").close()
        open(os.path.join(self.tmp.name, "libggml-cuda.so"), "w").close()
        code, out, _ = self._run([
            "run", self.path, "--target", HW, "--binary", binary, "--dry-run",
        ])
        self.assertEqual(code, 0)
        self.assertIn("--n-cpu-moe 2", out)
        self.assertIn("-ctk q8_0", out)
        self.assertIn(self.path, out)

    def test_requires_target_or_manifest(self):
        code, _, err = self._run(["run", self.path, "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("--target", err)

    def test_runs_a_binary_and_reports_throughput(self):
        binary = _fake_llama_cli(self.tmp.name)
        report_path = os.path.join(self.tmp.name, "run.json")
        code, out, _ = self._run([
            "run", self.path, "--target", HW, "--binary", binary,
            "--json", report_path,
        ])
        self.assertEqual(code, 0)
        self.assertIn("generation    7.50 t/s", out)
        with open(report_path) as f:
            report = json.load(f)
        self.assertTrue(report["ok"])
        self.assertEqual(report["generation_tokens_per_s"], 7.5)

    def test_trace_arms_the_prefetch_patch(self):
        # A binary that contains the patch's env var is detected as supporting it,
        # and the runner sets it plus writes the learned table.
        binary = os.path.join(self.tmp.name, "patched-cli")
        with open(binary, "w") as f:
            f.write("#!/bin/sh\n")
            f.write('echo "PREFETCH=$LLAMA_MOE_PREFETCH"\n')
            f.write('echo "[ Prompt: 10.0 t/s | Generation: 7.5 t/s ]"\n')
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
        self.assertTrue(supports_prefetch(binary))

        trace = os.path.join(self.tmp.name, "trace.jsonl")
        save_trace(trace, synth_trace(30, n_layers=2, n_experts=4, top_k=2, seed=0))
        outdir = os.path.join(self.tmp.name, "pf")
        report_path = os.path.join(self.tmp.name, "run.json")
        code, _, err = self._run([
            "run", self.path, "--target", HW, "--binary", binary,
            "--trace", trace, "--prefetch-out", outdir, "--json", report_path,
        ])
        self.assertEqual(code, 0)
        self.assertNotIn("no MoE prefetch support", err)
        with open(report_path) as f:
            report = json.load(f)
        table = report["env"]["LLAMA_MOE_PREFETCH"]
        self.assertTrue(table.endswith("prefetch.txt"))
        self.assertTrue(os.path.exists(table))

    def test_unpatched_binary_warns_about_a_useless_table(self):
        binary = _fake_llama_cli(self.tmp.name)  # does not mention the env var
        self.assertFalse(supports_prefetch(binary))
        trace = os.path.join(self.tmp.name, "trace.jsonl")
        save_trace(trace, synth_trace(10, n_layers=2, n_experts=4, top_k=2, seed=0))
        outdir = os.path.join(self.tmp.name, "pf")
        code, _, err = self._run([
            "run", self.path, "--target", HW, "--binary", binary,
            "--trace", trace, "--prefetch-out", outdir,
        ])
        self.assertEqual(code, 0)
        self.assertIn("no MoE prefetch support", err)

    def test_compare_runs_baseline_and_manifest(self):
        binary = _fake_llama_cli(self.tmp.name)
        code, out, _ = self._run([
            "run", self.path, "--target", HW, "--binary", binary, "--compare",
        ])
        self.assertEqual(code, 0)
        self.assertIn("baseline", out)
        self.assertIn("manifest cmd", out)
        self.assertIn("speedup", out)
        # baseline has no manifest flags, the manifest command does
        self.assertIn("-ngl 0", out)

    def test_compare_speedup_reflects_the_two_runs(self):
        # The fake binary reports 5 t/s without the prefetch env and 10 t/s with it,
        # so the speedup column must read 2.00x — proving the two runs are distinct
        # and the env reaches the manifest run only.
        binary = os.path.join(self.tmp.name, "vary-cli")
        with open(binary, "w") as f:
            f.write("#!/bin/sh\n")
            f.write('if [ -n "$LLAMA_MOE_PREFETCH" ]; then G=10.0; else G=5.0; fi\n')
            f.write('echo "[ Prompt: 1.0 t/s | Generation: ${G} t/s ]"\n')
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
        trace = os.path.join(self.tmp.name, "trace.jsonl")
        save_trace(trace, synth_trace(20, n_layers=2, n_experts=4, top_k=2, seed=0))
        outdir = os.path.join(self.tmp.name, "pf")
        code, out, _ = self._run([
            "run", self.path, "--target", HW, "--binary", binary, "--compare",
            "--trace", trace, "--prefetch-out", outdir,
        ])
        self.assertEqual(code, 0)
        self.assertIn("2.00x", out)

    def test_compare_banner_when_one_side_dies(self):
        # A binary that dies only on the manifest side (the prefetch env is set
        # there and nowhere else) is the shape of a real failure: the baseline
        # numbers still print, so a table full of plausible cells can hide the
        # fact that no comparison happened. The banner has to say so outright,
        # and the speedup line must not be computed.
        binary = os.path.join(self.tmp.name, "die-on-manifest-cli")
        with open(binary, "w") as f:
            f.write("#!/bin/sh\n")
            f.write('if [ -n "$LLAMA_MOE_PREFETCH" ]; then exit 137; fi\n')
            f.write('echo "[ Prompt: 1.0 t/s | Generation: 5.0 t/s ]"\n')
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
        trace = os.path.join(self.tmp.name, "trace.jsonl")
        save_trace(trace, synth_trace(20, n_layers=2, n_experts=4, top_k=2, seed=0))
        outdir = os.path.join(self.tmp.name, "pf")
        code, out, _ = self._run([
            "run", self.path, "--target", HW, "--binary", binary, "--compare",
            "--trace", trace, "--prefetch-out", outdir,
        ])
        self.assertEqual(code, 1)
        self.assertIn("COMPARISON INVALID", out)
        self.assertIn("manifest did not complete", out)
        # the headline number is suppressed, not merely footnoted
        self.assertIn("NOT COMPUTED", out)
        self.assertNotIn("speedup: 0.00x", out)

    def test_compare_no_banner_when_both_succeed(self):
        # The banner must stay silent on a good run, or it stops meaning anything.
        binary = _fake_llama_cli(self.tmp.name)
        code, out, _ = self._run([
            "run", self.path, "--target", HW, "--binary", binary, "--compare",
        ])
        self.assertEqual(code, 0)
        self.assertNotIn("COMPARISON INVALID", out)
        self.assertIn("generation speedup: 1.00x", out)

    def test_a_failing_binary_changes_the_exit_code(self):
        binary = os.path.join(self.tmp.name, "bad-cli")
        with open(binary, "w") as f:
            f.write("#!/bin/sh\nexit 3\n")
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
        code, _, _ = self._run([
            "run", self.path, "--target", HW, "--binary", binary,
        ])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
