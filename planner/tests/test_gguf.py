import os
import tempfile
import unittest

from tests._fixture import (
    make_gguf,
    make_gguf_with_bogus_counts,
    make_gguf_with_corrupt_string_length,
    make_gguf_with_data,
    make_gguf_with_nested_arrays,
    make_gguf_with_oversized_array,
    make_unknown_quant_gguf,
)
from vipera.gguf import (
    _BLOCK,
    GGMLType,
    GGUFError,
    TensorInfo,
    UnknownQuantType,
    read_gguf,
)


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


class TestUnknownQuantType(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_n_bytes_refuses_an_unknown_type(self):
        t = TensorInfo("blk.0.ffn_gate_exps.weight", (256, 64, 4), ggml_type=999, offset=0)
        with self.assertRaises(UnknownQuantType):
            _ = t.n_bytes

    def test_read_rejects_an_unknown_type(self):
        # The old reader sized this as fp16 with no warning, i.e. a wrong answer.
        path = os.path.join(self.tmp.name, "mxfp4.gguf")
        make_unknown_quant_gguf(path, ggml_type=999)
        with self.assertRaises(UnknownQuantType):
            read_gguf(path)

    def test_read_rejects_an_unknown_type_even_with_no_data(self):
        # A descriptor-only file must still fail: without a block size the bytes lie.
        path = os.path.join(self.tmp.name, "meta-only.gguf")
        make_unknown_quant_gguf(path, ggml_type=39)  # e.g. a newer type vipera lacks
        with self.assertRaises(GGUFError):
            read_gguf(path)

    def test_block_table_covers_every_known_type(self):
        # Guards the table against an enum gaining a member without a block size.
        missing = [m.name for m in GGMLType if m not in _BLOCK]
        self.assertEqual(missing, [])


class TestMalformedHeader(unittest.TestCase):
    """Every length and count in a header is corruption-controlled: none may crash.

    These used to raise MemoryError (a 2^62 string length went straight to
    `f.read(n)`) and RecursionError (a corrupt element type recursed until Python's
    own limit), i.e. the reader crashed instead of reporting a bad file.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _path(self, name: str) -> str:
        return os.path.join(self.tmp.name, name)

    def test_corrupt_string_length_is_rejected_before_allocating(self):
        for length in (2 ** 40, 2 ** 62):
            with self.subTest(length=length):
                path = self._path(f"str{length}.gguf")
                make_gguf_with_corrupt_string_length(path, length=length)
                with self.assertRaises(GGUFError) as ctx:
                    read_gguf(path)
                self.assertIn("unexpected EOF", str(ctx.exception))

    def test_deeply_nested_array_is_rejected(self):
        path = self._path("nested.gguf")
        make_gguf_with_nested_arrays(path, depth=3000)
        with self.assertRaises(GGUFError) as ctx:
            read_gguf(path)
        self.assertIn("nested", str(ctx.exception))

    def test_legal_nesting_still_parses(self):
        # The depth cap must not reject a header a real writer could emit.
        path = self._path("shallow.gguf")
        make_gguf_with_nested_arrays(path, depth=2)
        self.assertEqual(read_gguf(path).version, 3)

    def test_array_count_larger_than_the_file_is_rejected(self):
        path = self._path("count.gguf")
        make_gguf_with_oversized_array(path, count=2 ** 62)
        with self.assertRaises(GGUFError) as ctx:
            read_gguf(path)
        self.assertIn("cannot fit", str(ctx.exception))

    def test_header_counts_larger_than_the_file_are_rejected(self):
        for kwargs in ({"kv_count": 2 ** 62}, {"tensor_count": 2 ** 62}):
            with self.subTest(**kwargs):
                path = self._path(f"hdr{list(kwargs)[0]}.gguf")
                make_gguf_with_bogus_counts(path, **kwargs)
                with self.assertRaises(GGUFError) as ctx:
                    read_gguf(path)
                self.assertIn("cannot fit", str(ctx.exception))


class TestExtentValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_short_data_section_is_rejected(self):
        # The tensor claims 2048 F32 values (8192 bytes) but only 100 are present.
        path = os.path.join(self.tmp.name, "half.gguf")
        make_gguf_with_data(path, data_bytes=100, tensor_elements=2048)
        with self.assertRaises(GGUFError) as ctx:
            read_gguf(path)
        self.assertIn("truncated", str(ctx.exception))

    def test_complete_data_section_is_accepted(self):
        path = os.path.join(self.tmp.name, "whole.gguf")
        make_gguf_with_data(path, data_bytes=2048 * 4, tensor_elements=2048)
        g = read_gguf(path)
        self.assertEqual(g.tensor("blk.0.attn_q.weight").n_bytes, 2048 * 4)
        self.assertGreater(g.data_offset, 0)

    def test_descriptor_only_file_is_allowed(self):
        # No data region at all: offsets cannot be checked, so it must not raise.
        path = os.path.join(self.tmp.name, "tiny.gguf")
        make_gguf(path, block_count=2, n_experts=4)
        g = read_gguf(path)
        self.assertLessEqual(g.file_size, g.data_offset)


if __name__ == "__main__":
    unittest.main()
