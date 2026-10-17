import json

import pytest

from vipera import config as cfg_mod


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIPERA_HOME", str(tmp_path))
    monkeypatch.setattr(cfg_mod, "VIPERA_HOME", tmp_path)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg_mod, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(cfg_mod, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", tmp_path / "config.json")
    return tmp_path


def test_load_defaults_when_missing(isolated_home):
    cfg = cfg_mod.load_config()
    assert cfg["host"] == "127.0.0.1"
    assert cfg["port"] == 8686
    assert cfg["ctx"] == 8192


def test_save_and_reload(isolated_home):
    cfg = cfg_mod.load_config()
    cfg["port"] = 9000
    cfg_mod.save_config(cfg)

    assert (isolated_home / "config.json").exists()
    assert cfg_mod.load_config()["port"] == 9000


def test_type_coercion(isolated_home):
    (isolated_home / "config.json").write_text(json.dumps({
        "port": "9999",
        "temperature": "0.3",
        "flash_attn": "false",
    }))
    cfg = cfg_mod.load_config()
    assert cfg["port"] == 9999
    assert cfg["temperature"] == pytest.approx(0.3)
    assert cfg["flash_attn"] is False


def test_corrupt_config_falls_back(isolated_home):
    (isolated_home / "config.json").write_text("not json {")
    assert cfg_mod.load_config()["port"] == 8686
