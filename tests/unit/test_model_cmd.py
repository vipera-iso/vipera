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


def test_import_downloads_a_url_reference(isolated_home, monkeypatch):
    from vipera.sources import url as url_source

    models = isolated_home / "models"
    ref = "https://example.com/tiny.gguf"
    seen = {}

    def fake_resolve(value):
        seen["resolve"] = value
        return [url_source.FileInfo(path="tiny.gguf", size_bytes=4)]

    def fake_download(value, file, dest):
        seen["download"] = (value, file.path, dest)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / file.path
        path.write_bytes(b"gguf")
        return path

    monkeypatch.setattr(url_source, "resolve", fake_resolve)
    monkeypatch.setattr(url_source, "download", fake_download)
    monkeypatch.setattr(model_cmd, "MODELS_DIR", models)
    monkeypatch.setattr(
        model_cmd, "_read_meta", lambda path, name=None: {"size_bytes": 4, "quant": "Q4_K_M"}
    )
    monkeypatch.setattr(model_cmd, "animated_loading_bar", lambda *a, **k: None)

    model_cmd.import_model(ref)

    assert seen["resolve"] == ref
    assert seen["download"] == (ref, "tiny.gguf", models)
    spec = reg_mod.get_model("tiny")
    assert spec is not None
    assert spec.source == "url"
    assert spec.source_ref == ref
    assert spec.format == "gguf"


def test_add_is_an_alias_for_import(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        model_cmd, "import_model", lambda src, name=None: seen.update(src=src, name=name)
    )

    model_cmd.add_model_cmd("ollama:qwen2.5:7b", name="qwen")

    assert seen == {"src": "ollama:qwen2.5:7b", "name": "qwen"}


def test_import_from_the_ollama_store_makes_no_copy(isolated_home, monkeypatch, tmp_path):
    import hashlib
    import json

    store = tmp_path / "ollama-models"
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    payload = b"GGUF-payload"
    digest = hashlib.sha256(payload).hexdigest()
    (store / "blobs").mkdir(parents=True)
    blob = store / "blobs" / f"sha256-{digest}"
    blob.write_bytes(payload)
    manifest = store / "manifests-v2/ollama.com/library/qwen05/latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": f"sha256:{digest}",
                        "size": len(payload),
                        "from": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(model_cmd, "MODELS_DIR", isolated_home / "models")
    monkeypatch.setattr(
        model_cmd,
        "_read_meta",
        lambda path, name=None: {"size_bytes": len(payload), "quant": "Q4_K_M"},
    )
    monkeypatch.setattr(model_cmd, "animated_loading_bar", lambda *a, **k: None)

    model_cmd.import_model("ollama:qwen05")

    spec = reg_mod.get_model("qwen2.5-0.5b-instruct-q4_k_m")
    assert spec.source == "ollama"
    assert spec.source_ref == "qwen05"  # the ref the source understands
    assert spec.path.read_bytes() == payload
    # Linked to Ollama's blob: no second copy of the model on this disk.
    assert spec.path.stat().st_ino == blob.stat().st_ino


def test_import_accepts_a_local_prefix(isolated_home, monkeypatch):
    models = isolated_home / "models"
    src = isolated_home / "source.gguf"
    src.write_bytes(b"x")
    monkeypatch.setattr(model_cmd, "MODELS_DIR", models)
    monkeypatch.setattr(
        model_cmd, "_read_meta", lambda path, name=None: {"size_bytes": 1, "quant": "?"}
    )
    monkeypatch.setattr(model_cmd, "animated_loading_bar", lambda *a, **k: None)

    model_cmd.import_model(f"local:{src}")

    assert (models / "source.gguf").read_bytes() == b"x"
    spec = reg_mod.get_model("source")
    assert spec.source == "local"


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


def test_optimize_cmd_passes_tensor_type_overrides(monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr(
        model_cmd,
        "analyze_gguf",
        lambda path: {"metadata": {"file_size_bytes": 4_800_000_000, "quant": "Q4_K_M"}},
    )
    monkeypatch.setattr(
        model_cmd, "find_llama_tool", lambda tool, bin_dir=None: "/bin/llama-quantize"
    )
    monkeypatch.setattr(model_cmd, "tool_supports_flag", lambda binary, flag: True)
    monkeypatch.setattr(model_cmd, "_prepare_output", lambda value, fallback: fallback)
    seen = {}

    def fake_run(command, dry_run, cleanup=None):
        seen["command"] = command
        return True

    monkeypatch.setattr(model_cmd, "_run_tool", fake_run)

    model_cmd.optimize_cmd(
        path=str(model),
        target_size=None,
        target_quant="Q4_K_M",
        calibration_data=None,
        output=None,
        prune_layers=None,
        prior_weight=1,
        output_tensor=None,
        token_embedding=None,
        tensor_type=["ffn_.*_exps=Q3_K"],
        bin_dir=None,
        dry_run=True,
    )

    command = seen["command"]
    assert command[command.index("--tensor-type") + 1] == "ffn_.*_exps=Q3_K"
