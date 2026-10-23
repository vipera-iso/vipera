import json

import pytest

from vipera import config as cfg_mod
from vipera import registry as reg_mod


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


def test_add_and_get(isolated_home):
    reg_mod.add_model("tiny", "/tmp/tiny.gguf", {"quant": "Q4_K_M"})
    m = reg_mod.get_model("tiny")
    assert m is not None
    assert m.meta["quant"] == "Q4_K_M"
    assert m.alias == "tiny"
    assert m.source == "local"
    assert m.format == "gguf"


def test_add_records_where_the_file_came_from(isolated_home):
    spec = reg_mod.add_model(
        "hf-model",
        "/tmp/hf.gguf",
        source="hf",
        source_ref="Qwen/Qwen2.5-7B-Instruct-GGUF",
    )
    assert spec.source == "hf"
    assert spec.source_ref == "Qwen/Qwen2.5-7B-Instruct-GGUF"


def test_list(isolated_home):
    reg_mod.add_model("a", "/tmp/a.gguf")
    reg_mod.add_model("b", "/tmp/b.gguf")
    assert {m.alias for m in reg_mod.list_models()} == {"a", "b"}


def test_remove(isolated_home):
    reg_mod.add_model("gone", "/tmp/gone.gguf")
    assert reg_mod.remove_model("gone") is True
    assert reg_mod.get_model("gone") is None
    assert reg_mod.remove_model("gone") is False


def test_writes_the_versioned_shape(isolated_home):
    reg_mod.add_model("a", "/tmp/a.gguf")

    raw = json.loads((isolated_home / "registry.json").read_text())
    assert raw["_schema"] == reg_mod.SCHEMA_VERSION
    assert "a" in raw["models"]
    assert raw["models"]["a"]["path"].endswith("a.gguf")


def test_reads_a_1x_flat_registry_without_losing_models(isolated_home):
    legacy = {
        "old-model": {
            "name": "old-model",
            "path": "/tmp/old.gguf",
            "added": "2026-01-01T00:00:00Z",
            "meta": {"quant": "Q4_K_M", "n_layers": 8},
        }
    }
    (isolated_home / "registry.json").write_text(json.dumps(legacy))

    specs = reg_mod.list_models()
    assert [s.alias for s in specs] == ["old-model"]
    assert specs[0].source == "local"  # 1.x had no source field
    assert specs[0].format == "gguf"
    assert specs[0].meta["n_layers"] == 8

    # The next write upgrades the file and keeps the migrated entry.
    reg_mod.add_model("new-model", "/tmp/new.gguf")
    raw = json.loads((isolated_home / "registry.json").read_text())
    assert raw["_schema"] == 2
    assert set(raw["models"]) == {"old-model", "new-model"}


def test_entry_without_a_path_is_skipped(isolated_home):
    (isolated_home / "registry.json").write_text(json.dumps({"broken": {"meta": {}}}))

    assert reg_mod.list_models() == []


def test_corrupt_registry_is_backed_up(isolated_home, capsys):
    # Without the backup, the next add_model() would save a one-entry registry
    # over the user's file and every other model would vanish from the list.
    (isolated_home / "registry.json").write_text("{oops")

    reg_mod.add_model("only", "/tmp/only.gguf")

    assert [m.alias for m in reg_mod.list_models()] == ["only"]
    assert (isolated_home / "registry.json.corrupt").read_text() == "{oops"
    assert "unreadable" in capsys.readouterr().err


def test_registry_of_wrong_shape_is_treated_as_empty(isolated_home):
    (isolated_home / "registry.json").write_text("[1, 2, 3]")

    assert reg_mod.list_models() == []
    assert (isolated_home / "registry.json.corrupt").read_text() == "[1, 2, 3]"
