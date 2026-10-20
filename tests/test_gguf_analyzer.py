import sys
import types

import pytest

from vipera.commands import model_cmd
from vipera.gguf_analyzer import (
    GGUFAnalysisError,
    analyze_gguf,
    analyze_gguf_structure,
)


class FakeField:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class FakeTensor:
    def __init__(self, name, tensor_type, shape, n_bytes, n_elements=0):
        self.name = name
        self.tensor_type = tensor_type
        self.shape = shape
        self.n_bytes = n_bytes
        self.n_elements = n_elements


class FakeReader:
    def __init__(self):
        self.fields = {
            "general.architecture": FakeField("llama"),
            "general.name": FakeField("tiny-test"),
            "llama.block_count": FakeField(2),
            "llama.context_length": FakeField(4096),
            "llama.embedding_length": FakeField(128),
        }
        self.tensors = [
            FakeTensor("token_embd.weight", 1, [128, 100], 500, 12_800),
            FakeTensor("blk.0.attn_v.weight", 10, [128, 128], 100, 16_384),
            FakeTensor("blk.0.ffn_down.weight", 14, [128, 256], 200, 32_768),
            FakeTensor("blk.0.ffn_down_exps.weight", 12, [128, 256, 4], 300, 131_072),
            FakeTensor("blk.1.attn_q.weight", 12, [128, 128], 400, 16_384),
            FakeTensor("output_norm.weight", 0, [128], 50, 128),
        ]


@pytest.fixture
def mocked_gguf(monkeypatch):
    reader = FakeReader()
    module = types.ModuleType("gguf")
    module.GGUFReader = lambda path: reader
    monkeypatch.setitem(sys.modules, "gguf", module)
    return reader


def test_analyze_metadata_and_layout(tmp_path, mocked_gguf):
    path = tmp_path / "model-q2_k.gguf"
    path.write_bytes(b"")

    result = analyze_gguf(str(path))

    assert result["metadata"]["architecture"] == "llama"
    assert result["metadata"]["n_layers"] == 2
    assert result["metadata"]["context_length"] == 4096
    assert result["metadata"]["quant"] == "Q2_K"
    assert result["metadata"]["tensor_count"] == 6

    assert result["layer_distribution"][0] == 600
    assert result["layer_distribution"][1] == 400
    assert result["layer_distribution"][None] == 550
    assert result["category_sizes"]["expert"] == 300
    assert result["category_sizes"]["attention"] == 500
    assert result["category_sizes"]["norm"] == 50

    by_name = {tensor["name"]: tensor for tensor in result["tensor_map"]}
    assert by_name["blk.0.ffn_down_exps.weight"]["category"] == "expert"
    assert by_name["blk.0.ffn_down_exps.weight"]["layer"] == 0
    assert by_name["token_embd.weight"]["category"] == "embedding"
    assert by_name["output_norm.weight"]["category"] == "norm"


def test_analyze_warns_about_low_quant_critical_tensors(tmp_path, mocked_gguf):
    path = tmp_path / "model-q2_k.gguf"
    path.write_bytes(b"")

    result = analyze_gguf(str(path))

    warned = {warning["tensor"] for warning in result["warnings"]}
    assert "blk.0.attn_v.weight" in warned
    assert "blk.0.ffn_down.weight" not in warned


def test_analyze_requires_optional_gguf_package(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"")
    monkeypatch.setitem(sys.modules, "gguf", None)

    with pytest.raises(GGUFAnalysisError, match="gguf is not installed"):
        analyze_gguf(str(path))


def test_analyze_gguf_structure_is_an_alias():
    assert analyze_gguf_structure is analyze_gguf


def test_analyze_rejects_missing_file(tmp_path):
    with pytest.raises(GGUFAnalysisError, match="file not found"):
        analyze_gguf(str(tmp_path / "missing.gguf"))


def test_read_meta_records_size_of_one_repeating_layer(monkeypatch, tmp_path):
    """The planner budgets VRAM per layer from this, not from size/n_layers."""
    analysis = {
        "metadata": {
            "n_layers": 2,
            "n_params": 0,
            "context_length": 2048,
            "architecture": "llama",
            "embedding_length": 128,
            "n_heads": 4,
            "n_kv_heads": 2,
            "head_dim": 32,
            "dominant_quant": "Q4_K_M",
        },
        # None groups token_embd/output.weight: never offloaded with a layer.
        "layer_distribution": {None: 500, 0: 100, 1: 300},
    }
    path = tmp_path / "model-q4_k_m.gguf"
    path.write_bytes(b"x" * 10)
    monkeypatch.setattr(
        "vipera.gguf_analyzer.analyze_gguf", lambda p: analysis
    )

    meta = model_cmd._read_meta(path)

    assert meta["layer_bytes"] == 200  # mean of layers 0 and 1
    assert meta["n_layers"] == 2


def test_read_meta_without_layer_info_leaves_layer_bytes_zero(monkeypatch, tmp_path):
    analysis = {"metadata": {"n_layers": 2}, "layer_distribution": {}}
    path = tmp_path / "model.gguf"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        "vipera.gguf_analyzer.analyze_gguf", lambda p: analysis
    )

    assert model_cmd._read_meta(path)["layer_bytes"] == 0


def test_inspect_cli_renders_summary(monkeypatch, capsys, tmp_path):
    result = {
        "path": str(tmp_path / "model.gguf"),
        "metadata": {
            "architecture": "llama",
            "name": "tiny-test",
            "n_layers": 1,
            "context_length": 2048,
            "embedding_length": 128,
            "quant": "Q4_K_M",
            "dominant_quant": "Q2_K",
            "tensor_count": 1,
            "file_size_bytes": 600,
        },
        "tensor_map": [
            {
                "name": "blk.0.attn_v.weight",
                "shape": [2, 2],
                "quant": "Q2_K",
                "category": "attention",
                "layer": 0,
                "n_elements": 4,
                "n_bytes": 600,
            }
        ],
        "layer_distribution": {0: 600},
        "category_sizes": {"attention": 600},
        "quant_sizes": {"Q2_K": 600},
        "quant_counts": {"Q2_K": 1},
        "warnings": [
            {
                "tensor": "blk.0.attn_v.weight",
                "quant": "Q2_K",
                "category": "attention",
                "message": "too low",
            }
        ],
    }
    monkeypatch.setattr(model_cmd, "analyze_gguf", lambda path: result)

    model_cmd.inspect_model(str(tmp_path / "model.gguf"), tensors=False)
    output = capsys.readouterr().out

    assert "GGUF ANATOMY" in output
    assert "llama" in output
    assert "Q2_K" in output
    assert "QUALITY WARNINGS" in output
