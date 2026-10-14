import os
import unittest

from vipera.arch import Architecture
from vipera.longcontext import (
    DEFAULT_SINKS,
    kv_bytes_per_token,
    plan_long_context,
    resident_context_cap,
)
from vipera.run import RunFlags, build_command

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(REPO_ROOT, "arch", "deepseek-v2-lite.toml")

MIB = 1024 * 1024
GIB = 1024 * MIB


def _arch():
    return Architecture.from_toml(ARCH)


class TestKvBudget(unittest.TestCase):
    def test_per_token_slope_is_positive_and_scales_with_bits(self):
        arch = _arch()
        at16 = kv_bytes_per_token(arch, kv_bits=16)
        at8 = kv_bytes_per_token(arch, kv_bits=8)
        self.assertGreater(at16, 0)
        # Half the bits per parameter => half the bytes per token.
        self.assertEqual(at8, at16 // 2)

    def test_cap_grows_with_available_ram(self):
        arch = _arch()
        small = resident_context_cap(arch=arch, model_bytes=1 * GIB, ram_bytes=2 * GIB)
        large = resident_context_cap(arch=arch, model_bytes=1 * GIB, ram_bytes=8 * GIB)
        self.assertGreater(large, small)

    def test_cap_is_zero_when_model_leaves_no_budget(self):
        arch = _arch()
        cap = resident_context_cap(arch=arch, model_bytes=16 * GIB, ram_bytes=16 * GIB)
        self.assertEqual(cap, 0)


class TestPlanLongContext(unittest.TestCase):
    def test_within_cap_no_sliding(self):
        arch = _arch()
        p = plan_long_context(
            arch=arch, model_bytes=1 * GIB, ram_bytes=8 * GIB,
            requested_context=512, stream=False,
        )
        self.assertEqual(p.window, 512)
        self.assertFalse(p.context_shift)
        self.assertEqual(p.sinks, 0)
        self.assertFalse(p.slides)

    def test_beyond_cap_clamps_window_and_slides(self):
        arch = _arch()
        model_bytes, ram_bytes = 1 * GIB, 3 * GIB
        cap = resident_context_cap(arch=arch, model_bytes=model_bytes, ram_bytes=ram_bytes)
        self.assertGreater(cap, 0)
        p = plan_long_context(
            arch=arch, model_bytes=model_bytes, ram_bytes=ram_bytes,
            requested_context=cap + 4096, stream=True,
        )
        self.assertEqual(p.window, cap)
        self.assertTrue(p.context_shift)
        self.assertTrue(p.slides)
        self.assertTrue(any("clamping" in w for w in p.warnings))

    def test_stream_uses_sinks_and_shift(self):
        arch = _arch()
        p = plan_long_context(
            arch=arch, model_bytes=1 * GIB, ram_bytes=8 * GIB,
            requested_context=4096, stream=True, sinks=DEFAULT_SINKS,
        )
        self.assertTrue(p.context_shift)
        self.assertEqual(p.sinks, DEFAULT_SINKS)

    def test_sinks_clamped_to_a_quarter_of_the_window(self):
        arch = _arch()
        p = plan_long_context(
            arch=arch, model_bytes=1 * GIB, ram_bytes=8 * GIB,
            requested_context=8, stream=True, sinks=4,
        )
        self.assertLessEqual(p.sinks, max(1, p.window // 4))

    def test_flags_include_keep_and_shift(self):
        arch = _arch()
        p = plan_long_context(
            arch=arch, model_bytes=1 * GIB, ram_bytes=8 * GIB,
            requested_context=1024, stream=True,
        )
        flags = p.to_flags(kv_bits=8)
        self.assertIn("-c", flags)
        self.assertIn("--keep", flags)
        self.assertIn("--context-shift", flags)
        self.assertIn("q8_0", flags)

    def test_bad_arguments_rejected(self):
        arch = _arch()
        with self.assertRaises(ValueError):
            plan_long_context(arch=arch, model_bytes=1, ram_bytes=1, requested_context=0)
        with self.assertRaises(ValueError):
            plan_long_context(arch=arch, model_bytes=1, ram_bytes=1,
                              requested_context=10, sinks=-1)


class TestStreamFlagsInCommand(unittest.TestCase):
    def test_keep_and_shift_emitted_when_set(self):
        flags = RunFlags(ngl=0, n_keep=4, context_shift=True)
        cmd = build_command("llama-cli", "m.gguf", flags,
                            context=512, n_predict=32, prompt="hi")
        self.assertIn("--keep", cmd)
        self.assertEqual(cmd[cmd.index("--keep") + 1], "4")
        self.assertIn("--context-shift", cmd)

    def test_absent_by_default(self):
        cmd = build_command("llama-cli", "m.gguf", RunFlags(ngl=0),
                            context=512, n_predict=32, prompt="hi")
        self.assertNotIn("--keep", cmd)
        self.assertNotIn("--context-shift", cmd)


if __name__ == "__main__":
    unittest.main()
