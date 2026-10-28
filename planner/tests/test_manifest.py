import os
import tempfile
import unittest

from tests._fixture import make_gguf
from vipera.analyze import analyze
from vipera.gguf import read_gguf
from vipera.manifest import MIB, Target, plan

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(REPO_ROOT, "hw", "gtx1650.toml")


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


class TestTargetProfile(unittest.TestCase):
    """The shipped profile has to describe the machine the docs measured."""

    def test_usable_ram_is_the_documented_available_ram(self):
        # README target table: 15 GiB total, ~10 GiB available, no swap. The profile used
        # to leave 13.5 GiB usable, which is RAM the machine does not have.
        t = Target.from_toml(HW)
        self.assertEqual(t.ram_bytes, 15 * 1024 * MIB)
        self.assertEqual(t.usable_ram, 10 * 1024 * MIB)
        self.assertEqual(t.usable_vram, (4096 - 512) * MIB)
        self.assertEqual(t.disk_bw_gbs, 8.0)  # measured, not the old assumed 2.0


class TestSpillPenalty(unittest.TestCase):
    """The spill warning is derived from the target's tiers, never asserted."""

    def setUp(self):
        self.inv = _inv()

    def _spilling(self, **kw) -> Target:
        # 168 KiB always-resident fits in 192 KiB + 64 KiB; the 512 KiB of experts cannot.
        return Target("tiny", vram_bytes=192 * 1024, ram_bytes=64 * 1024,
                      vram_headroom_bytes=0, ram_headroom_bytes=0, **kw)

    def _warnings(self, **kw) -> str:
        return " ".join(plan(self.inv, self._spilling(**kw)).warnings)

    def test_penalty_follows_the_declared_tiers(self):
        measured = self._warnings(ram_bw_gbs=44.0, disk_bw_gbs=8.0)
        self.assertIn("~5.5x", measured)
        # Same spill, a disk twice as fast: the warning has to move with it.
        faster = self._warnings(ram_bw_gbs=44.0, disk_bw_gbs=22.0)
        self.assertIn("~2.0x", faster)
        self.assertNotEqual(measured, faster)

    def test_20x_only_when_the_tiers_imply_it(self):
        # This used to print 20x for every target; now it may only appear when the target
        # really declares tiers that make a miss 20x dearer.
        self.assertIn("~20.0x", self._warnings(ram_bw_gbs=40.0, disk_bw_gbs=2.0))
        self.assertNotIn("~20.0x", self._warnings(ram_bw_gbs=44.0, disk_bw_gbs=8.0))

    def test_averaged_penalty_weights_the_per_miss_cost_by_the_spilled_share(self):
        result = plan(self.inv, self._spilling(ram_bw_gbs=44.0, disk_bw_gbs=8.0))
        spill = result.manifest["placement"]["expert_spill_disk_bytes"]
        share = spill / self.inv.expert_bytes
        w = " ".join(result.warnings)
        self.assertIn(f"{share:.0%} of the pool", w)
        self.assertIn(f"~{1 + share * (44.0 / 8.0 - 1):.1f}x", w)

    def test_target_without_disk_bandwidth_says_so_instead_of_dividing(self):
        result = plan(self.inv, self._spilling(ram_bw_gbs=44.0, disk_bw_gbs=0.0))
        self.assertTrue(any("no disk bandwidth" in x for x in result.warnings))
        self.assertTrue(any("spill" in x for x in result.warnings))
        self.assertTrue(result.runnable)


if __name__ == "__main__":
    unittest.main()
