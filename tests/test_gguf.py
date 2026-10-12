import os
import tempfile
import unittest

from tests._fixture import make_gguf
from vipera.gguf import GGMLType, GGUFError, read_gguf


class TestGGUFReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tiny.gguf")
        make_gguf(self.path, block_count=2, n_experts=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header(self):
        g = read_gguf(self.path)
        self.assertEqual(g.version, 3)
        self.assertEqual(g.architecture, "llama")
        self.assertEqual(g.metadata["llama.block_count"], 2)
        self.assertEqual(g.metadata["general.tags"], ["test", "moe"])

    def test_tensor_count(self):
        g = read_gguf(self.path)
        # token_embd once, then 4 tensors (attn_q, gate_inp, gate_shexp, gate_exps) per layer.
        self.assertEqual(len(g.tensors), 1 + 2 * 4)

    def test_tensor_shape_and_bytes(self):
        g = read_gguf(self.path)
        embd = g.tensor("token_embd.weight")
        self.assertIsNotNone(embd)
        self.assertEqual(embd.shape, (256, 8))
        self.assertEqual(embd.ggml_type, GGMLType.F32)
        self.assertEqual(embd.n_elements, 2048)
        self.assertEqual(embd.n_bytes, 256 * 8 * 4)

    def test_expert_tensor_is_three_dimensional(self):
        g = read_gguf(self.path)
        exps = g.tensor("blk.1.ffn_gate_exps.weight")
        self.assertEqual(len(exps.shape), 3)
        self.assertEqual(exps.shape[2], 4)

    def test_rejects_non_gguf(self):
        bad = os.path.join(self.tmp.name, "bad.bin")
        with open(bad, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 32)
        with self.assertRaises(GGUFError):
            read_gguf(bad)

    def test_rejects_truncated(self):
        trunc = os.path.join(self.tmp.name, "trunc.gguf")
        with open(self.path, "rb") as src, open(trunc, "wb") as dst:
            dst.write(src.read(12))
        with self.assertRaises(GGUFError):
            read_gguf(trunc)


if __name__ == "__main__":
    unittest.main()
