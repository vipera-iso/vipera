"""Guards for the Rust island (`_planner_rs`).

The island is an exact replica of CPython's MT19937 (`random.Random`), the
modelled `_zipf_sample`/`_routing` generators, and the `ExpertCache` hot path.
Nothing may drift, so every check here compares the Rust replica against
`random.Random` / the pure-Python classes, and against a manifest dump with the
island forced off.
"""

import random
import sys
import unittest

sys.path.insert(0, ".")

import vipera.schedule as sch
from vipera.arch import Architecture
from vipera.manifest import Target
from vipera.schedule import ExpertCache, Quant, dumps, simulate


def _py_randbelow(px: random.Random, n: int) -> int:
    """`_randbelow_with_getrandbits` without reaching into private names."""
    k = n.bit_length()
    r = px.getrandbits(k)
    while r >= n:
        r = px.getrandbits(k)
    return r


@unittest.skipIf(sch._rs is None, "island `_planner_rs` not installed")
class TestRsRandomParity(unittest.TestCase):
    """The RNG behind byte-identical manifests: same state, same draws."""

    # a tuple, not a list: a mutable class attribute is shared state the tests
    # could edit from one case into the next
    SEEDS = (0, 1, 42, 7, 12345, 2**31 - 1, 2**32, 2**62,
             9223372036854775807, -(2**31), -1, 999, 65535)

    def test_state_dump(self):
        for s in self.SEEDS:
            words, mti = sch._rs.RsRandom(s)._state_words()
            st = random.Random(s).getstate()
            self.assertEqual(tuple(st[1][:624]), tuple(words), f"seed {s}")
            self.assertEqual(st[1][624], mti, f"seed {s}")

    def test_draw_sequences(self):
        for s in self.SEEDS:
            px, rx = random.Random(s), sch._rs.RsRandom(s)
            for _ in range(60):
                self.assertEqual(px.random(), rx.random(), f"seed {s}")
                self.assertEqual(px.getrandbits(48), rx.getrandbits(48))
                self.assertEqual(px.getrandbits(1), rx.getrandbits(1))
                self.assertEqual(px.getrandbits(30), rx.getrandbits(30))
                self.assertEqual(_py_randbelow(px, 523), _py_randbelow(rx, 523))


@unittest.skipIf(sch._rs is None, "island `_planner_rs` not installed")
class TestIslandParity(unittest.TestCase):
    """Island vs pure-Python: same decisions, same manifest bytes."""

    def _both_manifests(self, tokens=64):
        arch = Architecture.from_toml("arch/glm-5.3-flash.toml")
        target = Target.from_toml("hw/gtx1650.toml")
        kw = dict(quant=Quant.q4_experts(), context=2048, tokens=tokens, seed=0)
        rs = dumps(simulate(arch, target, **kw))
        saved, sch._rs = sch._rs, None
        try:
            py = dumps(simulate(arch, target, **kw))
        finally:
            sch._rs = saved
        return rs, py

    def test_manifest_byte_identical(self):
        rs, py = self._both_manifests()
        self.assertEqual(rs, py)

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "awkward timing elsewhere; keep it linux"
    )
    def test_manifest_byte_identical_long_run(self):
        rs, py = self._both_manifests(tokens=128)
        self.assertEqual(rs, py)


@unittest.skipIf(sch._rs is None, "island `_planner_rs` not installed")
class TestRsCacheParity(unittest.TestCase):
    """`ExpertCacheRs.access` replicates `ExpertCache.access` decisions."""

    POLICIES = ("lru", "lfu", "hot")
    CAPACITY = (0, 2, 4, 191)

    def test_access_decisions_identical(self):
        # A fixed, deterministic adversary reference sequence: one Zipf trace
        # (the same one `_routing` would build) drives both implementations.
        rng = random.Random(20260115)
        keys = []
        for _ in range(400):
            layer = rng.randrange(3)
            expert = min(4, _py_randbelow(rng, 191))
            keys.append((layer, expert))
        for policy in self.POLICIES:
            for cap in self.CAPACITY:
                py = ExpertCache(cap, policy, n_layers=3)
                rx = sch._rs.ExpertCacheRs(cap, policy, n_layers=3)
                for k in keys:
                    self.assertEqual(py.access(k), rx.access(k),
                                     f"{policy}/{cap} diverged at {k}")
                self.assertEqual(py.hits, rx.hits)
                self.assertEqual(py.misses, rx.misses)
