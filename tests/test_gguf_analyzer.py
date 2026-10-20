import struct
import sys
import types

import pytest

from vipera import gguf_analyzer as _analyzer
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
    monkeypatch.setattr("vipera.gguf_analyzer.analyze_gguf", lambda p: analysis)

    meta = model_cmd._read_meta(path)

    assert meta["layer_bytes"] == 200  # mean of layers 0 and 1
    assert meta["n_layers"] == 2


def test_read_meta_without_layer_info_leaves_layer_bytes_zero(monkeypatch, tmp_path):
    analysis = {"metadata": {"n_layers": 2}, "layer_distribution": {}}
    path = tmp_path / "model.gguf"
    path.write_bytes(b"x")
    monkeypatch.setattr("vipera.gguf_analyzer.analyze_gguf", lambda p: analysis)

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


# --- 2026 GGUF quant types (35, 39, 40, 41, 42) -----------------------------
#
# These build a real GGUF v3 file instead of faking GGUFReader, so the mapping
# is exercised through the same reader the CLI uses. gguf 0.19 knows 35/39/40/41
# and rejects 42; both paths are covered.


def _gguf_str(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path, tensor_type: int, *, name="blk.0.attn_v.weight", block=None):
    """Write a one-tensor GGUF v3 file with the given tensor type id."""
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    block_elems, block_bytes = block or GGML_QUANT_SIZES[GGMLQuantizationType(tensor_type)]
    dims = [block_elems, 4]

    kv = b""
    for key, value_type, value in (
        ("general.architecture", 8, _gguf_str("llama")),
        ("llama.block_count", 4, struct.pack("<I", 1)),
    ):
        kv += _gguf_str(key) + struct.pack("<I", value_type) + value

    info = _gguf_str(name) + struct.pack("<I", len(dims))
    info += b"".join(struct.pack("<Q", d) for d in dims)
    info += struct.pack("<I", tensor_type) + struct.pack("<Q", 0)

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 2)
    body = header + kv + info
    body += b"\x00" * ((-len(body)) % 32)
    path.write_bytes(body + b"\x00" * (4 * block_bytes))
    return block_bytes


@pytest.mark.parametrize(
    ("type_id", "expected"),
    [(35, "TQ2_0"), (39, "MXFP4"), (40, "NVFP4"), (41, "Q1_0"), (42, "Q2_0")],
)
def test_quant_id_names_cover_2026_types(type_id, expected):
    assert _analyzer._quant_name(type_id) == expected


def test_unknown_quant_rank_is_none_not_zero():
    # A new 4-bit type must not be flagged as "below Q4_K" just because its id
    # is missing from the table.
    assert _analyzer._quant_rank("TYPE_99") is None
    assert _analyzer._quant_rank("NVFP4") == pytest.approx(4.0)
    assert _analyzer._quant_rank("Q1_0") == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("type_id", "quant", "expected_bytes"),
    [(40, "NVFP4", 4 * 36), (41, "Q1_0", 4 * 18)],
)
def test_analyze_reads_new_quant_types(tmp_path, type_id, quant, expected_bytes):
    path = tmp_path / "model.gguf"
    block_bytes = _write_gguf(path, type_id)

    result = analyze_gguf(str(path))

    tensor = result["tensor_map"][0]
    assert tensor["quant"] == quant
    assert tensor["n_bytes"] == expected_bytes == 4 * block_bytes
    assert result["metadata"]["dominant_quant"] == quant
    assert result["metadata"]["quant"] == quant
    # attn_v at 4 bits or below is exactly what the warning is for.
    assert result["warnings"] and result["warnings"][0]["quant"] == quant


def test_analyze_reports_unsupported_quant_type_instead_of_crashing(tmp_path):
    # Q2_0 (42) is not in gguf 0.19: the reader cannot size it. The analyzer
    # must surface a GGUFAnalysisError, not a traceback or a bogus TYPE_42 row.
    path = tmp_path / "model.gguf"
    _write_gguf(path, 42, block=(128, 18))

    with pytest.raises(GGUFAnalysisError, match="failed to read"):
        analyze_gguf(str(path))


def test_filename_quant_detects_nvfp4_and_tq2_0(tmp_path):
    assert _analyzer._QUANT_RE.search("Model-NVFP4.gguf").group(1).upper() == "NVFP4"
    assert _analyzer._QUANT_RE.search("Model-TQ2_0.gguf").group(1).upper() == "TQ2_0"
