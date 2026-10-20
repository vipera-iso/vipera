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
    (isolated_home / "config.json").write_text(
        json.dumps(
            {
                "port": "9999",
                "temperature": "0.3",
                "flash_attn": "false",
            }
        )
    )
    cfg = cfg_mod.load_config()
    assert cfg["port"] == 9999
    assert cfg["temperature"] == pytest.approx(0.3)
    assert cfg["flash_attn"] is False


def test_corrupt_config_falls_back(isolated_home):
    (isolated_home / "config.json").write_text("not json {")
    assert cfg_mod.load_config()["port"] == 8686


def test_repeat_penalty_in_defaults():
    # _chat_kwargs, server.ChatRequest and Engine.chat all read this key, so it
    # must exist in the defaults or it silently never takes effect.
    assert cfg_mod.DEFAULT_CONFIG["repeat_penalty"] == pytest.approx(1.1)


def test_repeat_penalty_is_coerced(isolated_home):
    (isolated_home / "config.json").write_text(json.dumps({"repeat_penalty": "1.4"}))
    assert cfg_mod.load_config()["repeat_penalty"] == pytest.approx(1.4)


def test_coerce_resets_unknown_kv_cache():
    out = cfg_mod._coerce({"kv_cache": "q9_z"})
    assert out["kv_cache"] == cfg_mod.DEFAULT_CONFIG["kv_cache"]


def test_coerce_keeps_valid_kv_cache():
    assert cfg_mod._coerce({"kv_cache": "q4_0"})["kv_cache"] == "q4_0"


@pytest.mark.parametrize("kv", sorted(cfg_mod._KV_CACHE_TYPES))
def test_every_declared_kv_cache_type_is_accepted(kv):
    assert cfg_mod._coerce_strict({"kv_cache": kv})["kv_cache"] == kv


def test_coerce_strict_rejects_unknown_kv_cache():
    with pytest.raises(ValueError, match="kv_cache"):
        cfg_mod._coerce_strict({"kv_cache": "q9_z"})


@pytest.mark.parametrize("value", ["maybe", "truthy", "2", "nope"])
def test_coerce_strict_rejects_garbage_booleans(value):
    # Lenient loading treats anything unknown as False; a command writing the
    # config back must not silently flip a flag the user mistyped.
    with pytest.raises(ValueError, match="flash_attn"):
        cfg_mod._coerce_strict({"flash_attn": value})


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("off", False),
        ("0", False),
    ],
    ids=repr,
)
def test_coerce_strict_parses_booleans(value, expected):
    assert cfg_mod._coerce_strict({"flash_attn": value})["flash_attn"] is expected


def test_save_config_is_atomic(isolated_home):
    cfg_mod.save_config({"port": 9000})
    assert not (isolated_home / "config.json.tmp").exists()
    assert json.loads((isolated_home / "config.json").read_text())["port"] == 9000


def test_invalid_kv_cache_on_disk_is_repaired(isolated_home):
    # The user-facing path: a hand-edited config with a typo must not crash
    # startup, and must not silently hand an unknown type to the engine.
    (isolated_home / "config.json").write_text(json.dumps({"kv_cache": "q4_k_m"}))
    cfg = cfg_mod.load_config()
    assert cfg["kv_cache"] == cfg_mod.DEFAULT_CONFIG["kv_cache"]
