import os
import tempfile
import unittest

from vipera.launch import (
    LaunchPolicy,
    choose_policy,
    describe,
    read_available_ram_bytes,
    read_physical_cores,
)
from vipera.run import RunFlags, build_command

MIB = 1024 * 1024


class TestChoosePolicy(unittest.TestCase):
    def test_fits_and_memlock_covers_chooses_mlock(self):
        p = choose_policy(
            model_bytes=4 * 1024 * MIB,
            ram_bytes=12 * 1024 * MIB,
            memlock_bytes=float("inf"),
            cores=4,
            logical_cores=8,
        )
        self.assertEqual(p.load_mode, "mlock")
        self.assertEqual(p.threads, 4)
        self.assertEqual(p.warnings, [])

    def test_fits_but_memlock_short_chooses_dio(self):
        p = choose_policy(
            model_bytes=9880 * MIB,
            ram_bytes=11500 * MIB,
            memlock_bytes=8 * MIB,
            cores=4,
            logical_cores=8,
        )
        self.assertEqual(p.load_mode, "dio")
        self.assertTrue(any("memlock" in w for w in p.warnings))

    def test_does_not_fit_chooses_mmap_and_warns(self):
        p = choose_policy(
            model_bytes=18500 * MIB,
            ram_bytes=11500 * MIB,
            memlock_bytes=float("inf"),
            cores=4,
            logical_cores=8,
        )
        self.assertEqual(p.load_mode, "mmap")
        self.assertTrue(any("exceeds available RAM" in w for w in p.warnings))

    def test_threads_default_to_physical_cores_not_logical(self):
        p = choose_policy(
            model_bytes=1 * MIB, ram_bytes=1024 * MIB,
            memlock_bytes=float("inf"), cores=4, logical_cores=8,
        )
        self.assertEqual(p.threads, 4)

    def test_threads_fall_back_to_half_when_cores_unknown(self):
        p = choose_policy(
            model_bytes=1 * MIB, ram_bytes=1024 * MIB,
            memlock_bytes=float("inf"), cores=None, logical_cores=8,
        )
        self.assertEqual(p.threads, 4)

    def test_threads_never_exceed_logical(self):
        p = choose_policy(
            model_bytes=1 * MIB, ram_bytes=1024 * MIB,
            memlock_bytes=float("inf"), cores=16, logical_cores=8,
        )
        self.assertEqual(p.threads, 8)

    def test_boundary_at_fit_margin_is_treated_as_fitting(self):
        p = choose_policy(
            model_bytes=950 * MIB, ram_bytes=1000 * MIB,
            memlock_bytes=float("inf"), cores=4, logical_cores=4,
        )
        self.assertEqual(p.load_mode, "mlock")

    def test_to_flags_and_describe(self):
        p = LaunchPolicy(threads=4, load_mode="dio", reason="test")
        self.assertEqual(p.to_flags(), ["-lm", "dio"])
        self.assertIn("-lm dio", describe(p))


class TestReadHelpers(unittest.TestCase):
    def test_read_available_ram_bytes_parses_meminfo(self):
        with tempfile.NamedTemporaryFile("w", suffix="meminfo", delete=False) as f:
            f.write("MemTotal:       16000000 kB\n")
            f.write("MemAvailable:   11541744 kB\n")
            path = f.name
        try:
            self.assertEqual(read_available_ram_bytes(path), 11541744 * 1024)
        finally:
            os.unlink(path)

    def test_read_available_ram_bytes_missing_file(self):
        self.assertIsNone(read_available_ram_bytes("/nonexistent/meminfo"))

    def test_read_physical_cores_counts_distinct_pairs(self):
        with tempfile.NamedTemporaryFile("w", suffix="cpuinfo", delete=False) as f:
            # 4 physical cores, 8 logical threads.
            for core in range(4):
                for _ in range(2):
                    f.write("processor\t: 0\n")
                    f.write(f"physical id\t: 0\n")
                    f.write(f"core id\t\t: {core}\n\n")
            path = f.name
        try:
            self.assertEqual(read_physical_cores(path), 4)
        finally:
            os.unlink(path)


class TestBuildCommandLoadMode(unittest.TestCase):
    def _flags(self, load_mode):
        return RunFlags(
            ngl=0, n_cpu_moe=27, moe_cache_mib=1024,
            cache_type_k="q8_0", cache_type_v="q8_0", load_mode=load_mode,
        )

    def test_load_mode_emitted_when_set(self):
        cmd = build_command("llama-cli", "m.gguf", self._flags("dio"),
                            context=512, n_predict=32, prompt="hi", threads=4)
        self.assertIn("-lm", cmd)
        self.assertEqual(cmd[cmd.index("-lm") + 1], "dio")
        self.assertEqual(cmd[cmd.index("-t") + 1], "4")

    def test_load_mode_absent_by_default(self):
        cmd = build_command("llama-cli", "m.gguf", RunFlags(ngl=0),
                            context=512, n_predict=32, prompt="hi")
        self.assertNotIn("-lm", cmd)


if __name__ == "__main__":
    unittest.main()
