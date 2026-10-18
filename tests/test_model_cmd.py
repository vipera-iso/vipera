import pytest

from vipera import gguf_analyzer
from vipera.commands.model_cmd import _read_meta
from vipera.planner import _estimate_kv_cache_gb


def test_read_meta_reads_quant_from_original_filename(tmp_path):
    model = tmp_path / "tmp-model.gguf"
    model.write_bytes(b"")

    meta = _read_meta(model, "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")

    assert meta["quant"] == "Q4_K_M"
    assert meta["size_bytes"] == 0


def test_read_meta_always_exposes_kv_estimate_fields(tmp_path, monkeypatch):
    # The planner sizes the KV cache from these keys. If _read_meta stops
    # copying them, the planner silently drops back to the fixed 1.5 GB @ 8k
    # heuristic and the improved estimate never engages — with no error
    # anywhere. Checking key presence is not enough: the keys are also in the
    # initial dict as zeros, so assert the values actually arrive from the
    # analyzer.
    model = tmp_path / "tmp-model.gguf"
    model.write_bytes(b"")

    def fake_analyze(_path):
        return {"metadata": {
            "n_layers": 80,
            "n_params": 70_000_000_000,
            "context_length": 32768,
            "embedding_length": 8192,
            "architecture": "qwen2",
            "n_heads": 64,
            "n_kv_heads": 8,
            "head_dim": 128,
            "dominant_quant": "Q4_K_M",
        }}

    monkeypatch.setattr(gguf_analyzer, "analyze_gguf", fake_analyze)
    meta = _read_meta(model, "model-Q4_K_M.gguf")

    assert meta["n_layers"] == 80
    assert meta["embedding_length"] == 8192
    assert meta["n_heads"] == 64
    assert meta["n_kv_heads"] == 8
    assert meta["head_dim"] == 128

    # And these must be enough for the planner to leave the legacy heuristic.
    assert _estimate_kv_cache_gb(meta, 8192, "q8_0") != pytest.approx(1.5)
