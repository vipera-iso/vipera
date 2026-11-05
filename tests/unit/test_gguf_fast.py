"""The fast GGUF reader must produce the same index as the gguf package.

Two levels are checked: the reader's own fields/tensors, and the whole
`analyze_gguf` dict, which is what every caller actually consumes. The second is
the contract -- a faster reader that changes one number in `tensor_map` is a
regression no matter how fast it is.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vipera import gguf_analyzer
from vipera.gguf_fast import GGUFReadError, read_gguf

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")


def _write_model(path: Path) -> Path:
    """A real GGUF file (the package writer, so the bytes are canonical)."""
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name("tiny-fast-reader")
    w.add_block_count(2)
    w.add_context_length(4096)
    w.add_embedding_length(128)
    w.add_head_count([2, 2])
    w.add_head_count_kv([1, 1])
    w.add_key_value("general.parameter_count", 1234, gguf.GGUFValueType.UINT64)
    w.add_tokenizer_model("llama")
    # The vocabulary is what made the old reader slow; it must still be skipped
    # without breaking the cursor for everything that follows it.
    w.add_token_list([f"tok{i}" for i in range(2000)])
    w.add_token_merges([f"a b {i}" for i in range(500)])
    w.add_tensor("token_embd.weight", np.zeros((100, 128), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((128, 128), dtype=np.float32))
    w.add_tensor("blk.0.ffn_down_exps.weight", np.zeros((4, 128, 256), dtype=np.float32))
    w.add_tensor("blk.1.attn_v.weight", np.zeros((128, 128), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    return _write_model(tmp_path_factory.mktemp("gguf") / "tiny.gguf")


def test_tensor_table_matches_the_package(tiny_model):
    fast = read_gguf(str(tiny_model))
    slow = gguf.GGUFReader(str(tiny_model))
    assert [t.name for t in fast.tensors] == [t.name for t in slow.tensors]
    for got, want in zip(fast.tensors, slow.tensors, strict=True):
        assert list(got.shape) == [int(x) for x in want.shape]
        assert got.tensor_type == int(want.tensor_type)
        assert got.n_elements == int(want.n_elements)
        assert got.n_bytes == int(want.n_bytes)


def test_metadata_the_analyzer_reads_matches_the_package(tiny_model):
    fast = read_gguf(str(tiny_model)).fields
    slow = gguf.GGUFReader(str(tiny_model)).fields
    for key in (
        "general.architecture",
        "general.name",
        "general.parameter_count",
        "llama.block_count",
        "llama.context_length",
        "llama.embedding_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
    ):
        assert key in fast, key
        assert fast[key].contents() == slow[key].contents(), key
    # The vocabulary is deliberately not materialized.
    assert "tokenizer.ggml.tokens" not in fast


def test_analysis_dict_is_identical(tiny_model, monkeypatch):
    monkeypatch.setenv("VIPERA_GGUF_READER", "fast")
    fast = gguf_analyzer.analyze_gguf(str(tiny_model))
    monkeypatch.setenv("VIPERA_GGUF_READER", "pkg")
    slow = gguf_analyzer.analyze_gguf(str(tiny_model))
    assert fast == slow


def test_truncated_file_raises_and_fast_mode_does_not_fall_back(tiny_model, monkeypatch):
    # Cut inside the header: a file whose payload is short still parses here
    # (tensor data is never touched), so the truncated-header case is the one
    # that must fail loudly.
    cut = tiny_model.with_suffix(".cut.gguf")
    cut.write_bytes(tiny_model.read_bytes()[:20])
    with pytest.raises(GGUFReadError):
        read_gguf(str(cut))

    monkeypatch.setenv("VIPERA_GGUF_READER", "fast")
    with pytest.raises(gguf_analyzer.GGUFAnalysisError):
        gguf_analyzer.analyze_gguf(str(cut))


def test_default_mode_falls_back_for_files_the_fast_reader_rejects(monkeypatch):
    # A tiny non-GGUF file: the package reader's error surfaces as before, which
    # proves the default path did not swallow it.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
        fh.write(b"not a gguf file at all")
        path = fh.name
    try:
        monkeypatch.delenv("VIPERA_GGUF_READER", raising=False)
        with pytest.raises(gguf_analyzer.GGUFAnalysisError):
            gguf_analyzer.analyze_gguf(path)
    finally:
        os.unlink(path)


@pytest.mark.skipif(
    not list((Path.home() / ".vipera" / "models").glob("*.gguf")),
    reason="no installed model to compare against",
)
def test_real_model_analysis_is_identical(tmp_path, monkeypatch):
    # Smallest installed model: the parity claim must hold on real quant types,
    # not only on the F32 tensors the writer produces.
    model = min(
        (Path.home() / ".vipera" / "models").glob("*.gguf"), key=lambda p: p.stat().st_size
    )
    monkeypatch.setenv("VIPERA_GGUF_READER", "fast")
    fast = gguf_analyzer.analyze_gguf(str(model))
    monkeypatch.setenv("VIPERA_GGUF_READER", "pkg")
    slow = gguf_analyzer.analyze_gguf(str(model))
    assert fast == slow
    assert fast["metadata"]["tensor_count"] > 0


def _string_array(strings: list[str], *, wide: bool = True) -> bytes:
    """A GGUF array of strings: `[len][bytes]` per entry, no header."""
    out = bytearray()
    for s in strings:
        raw = s.encode()
        out += (len(raw).to_bytes(8, "little") if wide else len(raw).to_bytes(4, "little"))
        out += raw
    return bytes(out)


def test_rust_walk_agrees_with_the_python_walk():
    rs = pytest.importorskip("_gguf_rs")
    payload = _string_array(["a", "bb", "", "ccc" * 50])
    assert rs.scan_strings(payload, 0, 4, True) == len(payload)
    # v1 files use 32-bit length prefixes; the walk has to follow the same rule.
    narrow = _string_array(["a", "bb"], wide=False)
    assert rs.scan_strings(narrow, 0, 2, False) == len(narrow)
    # A start offset, as the reader passes it.
    assert rs.scan_strings(payload, len(_string_array(["a", "bb", ""])), 1, True) == len(payload)


def test_rust_walk_refuses_a_length_that_runs_past_the_buffer():
    rs = pytest.importorskip("_gguf_rs")
    bad = (10_000).to_bytes(8, "little") + b"short"
    with pytest.raises(ValueError):
        rs.scan_strings(bad, 0, 1, True)
    with pytest.raises(ValueError):
        rs.scan_strings(b"", 0, 1, True)


def test_island_is_used_when_installed(monkeypatch):
    """The fast path must actually be wired, not merely importable."""
    pytest.importorskip("_gguf_rs")
    monkeypatch.delenv("VIPERA_GGUF_FAST_PURE", raising=False)
    import importlib

    from vipera import gguf_fast

    importlib.reload(gguf_fast)
    try:
        assert gguf_fast._SCAN_STRINGS is not None
        monkeypatch.setenv("VIPERA_GGUF_FAST_PURE", "1")
        importlib.reload(gguf_fast)
        assert gguf_fast._SCAN_STRINGS is None
    finally:
        monkeypatch.delenv("VIPERA_GGUF_FAST_PURE", raising=False)
        importlib.reload(gguf_fast)
