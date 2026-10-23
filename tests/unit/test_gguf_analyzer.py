import struct
import sys
import types

import pytest

from vipera import gguf_analyzer as _analyzer
from vipera.commands import model_cmd
from vipera.gguf_analyzer import (
    GGUFAnalysisError,
    _attention_profile,
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


class FakeMoEReader(FakeReader):
    """Two layers with a router, a shared expert, and routed experts."""

    def __init__(self):
        super().__init__()
        self.tensors = [
            FakeTensor("blk.0.ffn_gate_inp.weight", 1, [128, 4], 20, 512),
            FakeTensor("blk.0.ffn_gate_shexp.weight", 1, [128, 256], 60, 32_768),
            FakeTensor("blk.0.ffn_gate_exps.weight", 1, [128, 256, 4], 300, 131_072),
            FakeTensor("blk.1.ffn_down_exps.weight", 1, [128, 256, 4], 400, 131_072),
            FakeTensor("blk.1.ffn_up_shexp.weight", 1, [128, 256], 70, 32_768),
        ]


@pytest.fixture
def mocked_moe_gguf(monkeypatch):
    reader = FakeMoEReader()
    module = types.ModuleType("gguf")
    module.GGUFReader = lambda path: reader
    monkeypatch.setitem(sys.modules, "gguf", module)
    return reader


def test_analyze_separates_routed_and_shared_experts(tmp_path, mocked_moe_gguf):
    path = tmp_path / "moe.gguf"
    path.write_bytes(b"")

    result = analyze_gguf(str(path))

    meta = result["metadata"]
    # Router (ffn_gate_inp) and shared experts (ffn_*_shexp) are always active
    # and must not be counted as offloadable routed experts.
    assert meta["expert_bytes"] == 700
    assert meta["shared_expert_bytes"] == 130
    assert meta["expert_layers"] == 2
    assert meta["expert_layer_bytes"] == [300, 400]


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


def test_read_meta_copies_expert_layout(monkeypatch, tmp_path):
    analysis = {
        "metadata": {
            "n_layers": 2,
            "expert_bytes": 700,
            "shared_expert_bytes": 130,
            "expert_layers": 2,
            "expert_layer_bytes": [300, 400],
        },
        "layer_distribution": {0: 100, 1: 100},
    }
    path = tmp_path / "moe.gguf"
    path.write_bytes(b"x")
    monkeypatch.setattr("vipera.gguf_analyzer.analyze_gguf", lambda p: analysis)

    meta = model_cmd._read_meta(path)

    assert meta["expert_bytes"] == 700
    assert meta["shared_expert_bytes"] == 130
    assert meta["expert_layers"] == 2
    assert meta["expert_layer_bytes"] == [300, 400]


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

    assert "gguf anatomy" in output
    assert "llama" in output
    assert "Q2_K" in output
    # The warnings block is a ruled note, not a titled panel: assert the line
    # it prints for the flagged tensor instead of a heading.
    assert "blk.0.attn_v.weight: Q2_K" in output


# --- 2026 GGUF quant types (35, 39, 40, 41, 42) -----------------------------
#
# These build a real GGUF v3 file instead of faking GGUFReader, so the mapping
# is exercised through the same reader the CLI uses. gguf 0.19 knows 35/39/40/41
# and rejects 42; both paths are covered.


def _gguf_str(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf_quant_table():
    """Return `(GGML_QUANT_SIZES, GGMLQuantizationType)` from the gguf package.

    `gguf` is an optional extra that CI does not install, so tests that build a
    real GGUF file skip instead of failing at import time.
    """
    constants = pytest.importorskip("gguf.constants", reason="the gguf extra is not installed")
    return constants.GGML_QUANT_SIZES, constants.GGMLQuantizationType


def _write_gguf(path, tensor_type: int, *, name="blk.0.attn_v.weight", block=None):
    """Write a one-tensor GGUF v3 file with the given tensor type id."""
    GGML_QUANT_SIZES, GGMLQuantizationType = _gguf_quant_table()

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


# --- hybrid attention detection (KDA/BCSA/Mamba style) -----------------------


def _write_gguf_multi(path, tensors, *, block_count, extra_kv=()):
    """Write a GGUF v3 file holding several tensors: [(name, type_id, dims)].

    ``extra_kv`` appends pre-packed metadata entries (key + type tag + value)
    after the two fixed ones, for fields the analyzer reads from GGUF metadata
    instead of from tensor names.
    """
    GGML_QUANT_SIZES, GGMLQuantizationType = _gguf_quant_table()

    infos, payload = b"", b""
    for name, type_id, dims in tensors:
        block_elems, block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType(type_id)]
        elements = 1
        for dim in dims:
            elements *= dim
        infos += _gguf_str(name) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", dim) for dim in dims)
        infos += struct.pack("<I", type_id) + struct.pack("<Q", len(payload))
        payload += b"\x00" * (elements // block_elems * block_bytes)

    kv = _gguf_str("general.architecture") + struct.pack("<I", 8) + _gguf_str("llama")
    kv += _gguf_str("llama.block_count") + struct.pack("<I", 4) + struct.pack("<I", block_count)
    for entry in extra_kv:
        kv += entry
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", 2 + len(extra_kv))
    body = header + kv + infos
    body += b"\x00" * ((-len(body)) % 32)
    path.write_bytes(body + payload)


def test_detects_hybrid_attention_layers(tmp_path):
    tensors = []
    for layer in range(3):
        # KDA/SSM layers: fixed-size state, no KV cache.
        tensors.append((f"blk.{layer}.ssm_conv1d.weight", 0, [4, 16]))
        tensors.append((f"blk.{layer}.ffn_down.weight", 0, [16, 16]))
    # One real attention layer.
    tensors += [
        ("blk.3.attn_k.weight", 0, [16, 16]),
        ("blk.3.attn_v.weight", 0, [16, 16]),
        ("blk.3.ffn_down.weight", 0, [16, 16]),
    ]
    path = tmp_path / "hybrid.gguf"
    _write_gguf_multi(path, tensors, block_count=4)

    meta = analyze_gguf(str(path))["metadata"]

    assert meta["hybrid_attention"] is True
    assert meta["kv_layers"] == 1
    assert meta["recurrent_layers"] == 3
    assert meta["n_layers"] == 4


def test_dense_model_is_not_marked_hybrid(tmp_path):
    tensors = []
    for layer in range(2):
        tensors.append((f"blk.{layer}.attn_k.weight", 0, [16, 16]))
        tensors.append((f"blk.{layer}.attn_v.weight", 0, [16, 16]))
    path = tmp_path / "dense.gguf"
    _write_gguf_multi(path, tensors, block_count=2)

    meta = analyze_gguf(str(path))["metadata"]

    assert meta["hybrid_attention"] is False
    assert meta["kv_layers"] == 2
    assert meta["recurrent_layers"] == 0


def test_conv_only_layers_mark_hybrid_attention(tmp_path):
    # LFM2.5-style stacks: most blocks are `shortconv.*` and hold no K/V
    # tensors at all, only a few blocks carry attention. Skipping the conv
    # layers leaves `recurrent_layers` empty, `hybrid` never trips and the
    # planner prices every layer as KV-bearing (3.00 GB instead of 0.19 GB).
    tensors = [
        ("blk.0.shortconv.conv.weight", 0, [16, 4]),
        ("blk.0.shortconv.in_proj.weight", 0, [16, 16]),
        ("blk.0.ffn_down.weight", 0, [16, 16]),
        ("blk.1.shortconv.out_proj.weight", 0, [16, 16]),
        ("blk.1.ffn_down.weight", 0, [16, 16]),
        ("blk.2.attn_k.weight", 0, [16, 16]),
        ("blk.2.attn_v.weight", 0, [16, 16]),
        ("blk.2.ffn_down.weight", 0, [16, 16]),
    ]
    path = tmp_path / "conv-hybrid.gguf"
    _write_gguf_multi(path, tensors, block_count=3)

    meta = analyze_gguf(str(path))["metadata"]

    assert meta["hybrid_attention"] is True
    assert meta["kv_layers"] == 1
    assert meta["recurrent_layers"] == 2
    assert meta["n_layers"] == 3


def test_recurrent_only_model_falls_back_to_dense_kv_budget(tmp_path):
    # No attention tensors at all: the split is not understood, so the planner
    # must keep the conservative dense estimate rather than assume zero KV.
    tensors = [("blk.0.ssm_conv1d.weight", 0, [4, 16])]
    path = tmp_path / "recurrent.gguf"
    _write_gguf_multi(path, tensors, block_count=1)

    meta = analyze_gguf(str(path))["metadata"]

    assert meta["hybrid_attention"] is False
    assert meta["kv_layers"] == 1


def test_attention_profile_handles_fused_and_mla_tensor_names():
    # Fused QKV (Qwen3-Next style), MLA (DeepSeek attn_k_b/attn_kv_a_mqa) and
    # KDA layers must land in the right bucket without an architecture table.
    names = [
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_down.weight",
        "blk.1.attn_k_b.weight",
        "blk.1.attn_kv_a_mqa.weight",
        "blk.1.ffn_down.weight",
        "blk.2.linear_attn.qkv.weight",
        "blk.2.linear_attn.conv1d.weight",
        "blk.2.ffn_down.weight",
    ]

    profile = _attention_profile(names, 3)

    assert profile["hybrid_attention"] is True
    assert profile["kv_layers"] == 2
    assert profile["recurrent_layers"] == 1


def test_head_count_kv_array_skips_zero_layers(tmp_path):
    # lfm2moe writes attention.head_count_kv per layer: 0 on the conv-only
    # layers, 8 on the attention ones. Reading the first entry reports 0 KV
    # heads, so the planner falls back to the full head count (32) and prices
    # the KV cache 4x too high.
    tensors = [
        ("blk.0.attn_k.weight", 0, [16, 16]),
        ("blk.0.attn_v.weight", 0, [16, 16]),
    ]
    head_count_kv = (
        _gguf_str("llama.attention.head_count_kv")
        + struct.pack("<I", 9)  # GGUF array
        + struct.pack("<I", 5)  # element type: UINT32
        + struct.pack("<Q", 4)  # element count
        + struct.pack("<4I", 0, 0, 8, 0)
    )
    path = tmp_path / "kv-heads.gguf"
    _write_gguf_multi(path, tensors, block_count=4, extra_kv=(head_count_kv,))

    meta = analyze_gguf(str(path))["metadata"]

    assert meta["n_kv_heads"] == 8
