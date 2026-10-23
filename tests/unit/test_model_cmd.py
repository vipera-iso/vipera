import pytest

from vipera import config as cfg_mod
from vipera import gguf_analyzer
from vipera import registry as reg_mod
from vipera.commands import model_cmd
from vipera.commands.model_cmd import _read_meta, recommend_quantization, rm_cmd
from vipera.planner import _estimate_kv_cache_gb


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIPERA_HOME", str(tmp_path))
    monkeypatch.setattr(cfg_mod, "VIPERA_HOME", tmp_path)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg_mod, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(cfg_mod, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(reg_mod, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def test_list_cmd_renders_model_specs(monkeypatch, capsys):
    from pathlib import Path

    from vipera.registry import ModelSpec

    monkeypatch.setattr(
        model_cmd,
        "list_models",
        lambda: [
            ModelSpec(
                alias="m",
                path=Path("/tmp/m.gguf"),
                meta={
                    "size_bytes": 1_500_000_000,
                    "quant": "Q4_K_M",
                    "n_layers": 4,
                    "context_length": 8192,
                },
            )
        ],
    )

    model_cmd.list_cmd()

    out = capsys.readouterr().out
    assert "m" in out
    assert "Q4_K_M" in out
    assert "8,192" in out


def test_recommend_command_works_on_an_already_quantized_model(monkeypatch, capsys):
    # The suggested command targets the model's current quant, which is by
    # definition already quantized: llama-quantize refuses without this flag.
    monkeypatch.setattr(
        model_cmd,
        "analyze_gguf",
        lambda path: {"metadata": {"quant": "Q2_K"}, "warnings": ["blk.0.ffn_down"]},
    )

    recommend_quantization("m.gguf")

    out = capsys.readouterr().out
    assert "vipera model quantize" in out
    assert "--allow-requantize" in out


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
        return {
            "metadata": {
                "n_layers": 80,
                "n_params": 70_000_000_000,
                "context_length": 32768,
                "embedding_length": 8192,
                "architecture": "qwen2",
                "n_heads": 64,
                "n_kv_heads": 8,
                "head_dim": 128,
                "dominant_quant": "Q4_K_M",
            }
        }

    monkeypatch.setattr(gguf_analyzer, "analyze_gguf", fake_analyze)
    meta = _read_meta(model, "model-Q4_K_M.gguf")

    assert meta["n_layers"] == 80
    assert meta["embedding_length"] == 8192
    assert meta["n_heads"] == 64
    assert meta["n_kv_heads"] == 8
    assert meta["head_dim"] == 128

    # And these must be enough for the planner to leave the legacy heuristic.
    assert _estimate_kv_cache_gb(meta, 8192, "q8_0") != pytest.approx(1.5)


def test_rm_keeps_a_file_another_alias_still_references(isolated_home):
    # `model search` twice for the same download yields two aliases, one file.
    f = isolated_home / "model.gguf"
    f.write_bytes(b"x")
    reg_mod.add_model("one", f)
    reg_mod.add_model("two", f)

    rm_cmd("two", yes=True)

    assert f.exists()  # alias "one" still points here
    assert reg_mod.get_model("two") is None

    rm_cmd("one", yes=True)

    assert not f.exists()  # last reference gone: the file is removed


def test_read_meta_propagates_hybrid_attention_split(tmp_path, monkeypatch):
    # The planner can only shrink the KV estimate if the hybrid layer split
    # survives the model -> planner hand-off in _read_meta.
    model = tmp_path / "hybrid.gguf"
    model.write_bytes(b"")

    def fake_analyze(_path):
        return {
            "metadata": {
                "n_layers": 72,
                "n_heads": 32,
                "n_kv_heads": 8,
                "head_dim": 128,
                "hybrid_attention": True,
                "kv_layers": 18,
                "recurrent_layers": 54,
            }
        }

    monkeypatch.setattr(gguf_analyzer, "analyze_gguf", fake_analyze)
    meta = _read_meta(model, "hybrid.gguf")

    assert meta["hybrid_attention"] is True
    assert meta["kv_layers"] == 18
    assert meta["recurrent_layers"] == 54
    # 18 attention layers, not 72: the estimate must follow the split.
    expected = 2 * 18 * 8 * 128 * 8192 * 1.0 / 1024**3
    assert _estimate_kv_cache_gb(meta, 8192, "q8_0") == pytest.approx(expected)
