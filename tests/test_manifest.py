import os
import tempfile
import unittest

from tests._fixture import make_gguf
from vipera.analyze import analyze
from vipera.gguf import read_gguf
from vipera.manifest import MIB, Target, plan


def _inv():
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "tiny.gguf")
    make_gguf(path, block_count=2, n_experts=4)
    inv = analyze(read_gguf(path))
    tmp.cleanup()
    return inv


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.inv = _inv()

    def test_roomy_target_keeps_everything_resident(self):
        t = Target("roomy", vram_bytes=4096 * MIB, ram_bytes=64 * 1024 * MIB)
        result = plan(self.inv, t)
        p = result.manifest["placement"]
        self.assertTrue(result.runnable)
        self.assertEqual(p["expert_spill_disk_bytes"], 0)
        self.assertEqual(p["expert_cache_fraction"], 1.0)
        self.assertEqual(result.warnings, [])

    def test_tiny_target_spills_and_warns(self):
        # 168 KiB always-resident fits in 192 KiB + 64 KiB, but the 512 KiB of
        # experts cannot, so the excess must spill to disk.
        t = Target(
            "tiny",
            vram_bytes=192 * 1024,
            ram_bytes=64 * 1024,
            vram_headroom_bytes=0,
            ram_headroom_bytes=0,
        )
        result = plan(self.inv, t)
        p = result.manifest["placement"]
        self.assertGreater(p["expert_spill_disk_bytes"], 0)
        self.assertTrue(any("spill" in w for w in result.warnings))
        self.assertTrue(result.runnable)  # the always-resident part still fits

    def test_unrunnable_when_resident_exceeds_memory(self):
        # 32 KiB each: less than the 168 KiB always-resident set, so nothing can run.
        t = Target(
            "absurd",
            vram_bytes=32 * 1024,
            ram_bytes=32 * 1024,
            vram_headroom_bytes=0,
            ram_headroom_bytes=0,
        )
        result = plan(self.inv, t)
        self.assertFalse(result.runnable)
        self.assertTrue(any("do not fit" in w for w in result.warnings))

    def test_manifest_has_the_contract_fields(self):
        t = Target("gtx1650", vram_bytes=4096 * MIB, ram_bytes=15360 * MIB, disk="nvme-dramless")
        m = plan(self.inv, t).manifest
        for key in ("vipera_version", "source", "target", "placement", "kv_cache"):
            self.assertIn(key, m)
        self.assertEqual(m["placement"]["expert_order"], "uniform")
        self.assertTrue(m["placement"]["expert_order_pending_trace"])

    def test_usable_memory_accounts_for_headroom(self):
        t = Target("t", vram_bytes=4096 * MIB, ram_bytes=4096 * MIB,
                   vram_headroom_bytes=512 * MIB, ram_headroom_bytes=1536 * MIB)
        self.assertEqual(t.usable_vram, (4096 - 512) * MIB)
        self.assertEqual(t.usable_ram, (4096 - 1536) * MIB)


if __name__ == "__main__":
    unittest.main()
