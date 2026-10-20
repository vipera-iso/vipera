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
    assert m["meta"]["quant"] == "Q4_K_M"


def test_list(isolated_home):
    reg_mod.add_model("a", "/tmp/a.gguf")
    reg_mod.add_model("b", "/tmp/b.gguf")
    names = {m["name"] for m in reg_mod.list_models()}
    assert names == {"a", "b"}


def test_remove(isolated_home):
    reg_mod.add_model("gone", "/tmp/gone.gguf")
    assert reg_mod.remove_model("gone") is True
    assert reg_mod.get_model("gone") is None
    assert reg_mod.remove_model("gone") is False


def test_corrupt_registry_is_backed_up(isolated_home, capsys):
    # Without the backup, the next add_model() would save a one-entry registry
    # over the user's file and every other model would vanish from the list.
    (isolated_home / "registry.json").write_text("{oops")

    reg_mod.add_model("only", "/tmp/only.gguf")

    assert [m["name"] for m in reg_mod.list_models()] == ["only"]
    assert (isolated_home / "registry.json.corrupt").read_text() == "{oops"
    assert "unreadable" in capsys.readouterr().err


def test_registry_of_wrong_shape_is_treated_as_empty(isolated_home):
    (isolated_home / "registry.json").write_text("[1, 2, 3]")

    assert reg_mod.list_models() == []
    assert (isolated_home / "registry.json.corrupt").read_text() == "[1, 2, 3]"
