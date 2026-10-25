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
            }
        )
    )
    cfg = cfg_mod.load_config()
    assert cfg["port"] == 9999
    assert cfg["temperature"] == pytest.approx(0.3)


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
def test_parse_bool_rejects_garbage(value):
    # Lenient loading warns and keeps the default; a command writing the config
    # back must not silently flip a flag the user mistyped either.
    with pytest.raises(ValueError, match="boolean"):
        cfg_mod._parse_bool(value)


def test_a_boolean_typo_warns_and_keeps_the_default(isolated_home, capsys):
    # "fasle" used to coerce to False without a word, silently disabling a
    # feature the user thought they had configured.
    (isolated_home / "config.json").write_text(json.dumps({"shaping": "fasle"}))

    cfg = cfg_mod.load_config()

    assert cfg["shaping"] is cfg_mod.DEFAULT_CONFIG["shaping"] is True
    assert "shaping" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("on", True),
        ("1", True),
        ("off", False),
        ("no", False),
        ("0", False),
    ],
    ids=repr,
)
def test_boolean_strings_are_still_coerced(isolated_home, value, expected):
    (isolated_home / "config.json").write_text(json.dumps({"embeddings": value}))
    assert cfg_mod.load_config()["embeddings"] is expected


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
def test_parse_bool_parses_booleans(value, expected):
    assert cfg_mod._parse_bool(value) is expected


def test_null_typed_value_falls_back_to_default(isolated_home):
    # A hand-edited `"ctx": null` survived the merge and crashed the planner
    # with a TypeError long after startup.
    (isolated_home / "config.json").write_text(json.dumps({"ctx": None, "port": None}))
    cfg = cfg_mod.load_config()
    assert cfg["ctx"] == 8192
    assert cfg["port"] == 8686


def test_invalid_typed_value_warns_and_falls_back(isolated_home, capsys):
    (isolated_home / "config.json").write_text(json.dumps({"ctx": "abc"}))
    cfg = cfg_mod.load_config()
    assert cfg["ctx"] == 8192
    assert "ctx" in capsys.readouterr().err


def test_flash_attn_is_not_a_config_key():
    # The planner derives flash attention from kv_cache; a config key that
    # nothing reads only makes the user think it takes effect.
    assert "flash_attn" not in cfg_mod.DEFAULT_CONFIG
    assert "flash_attn" not in cfg_mod._TYPE_MAP


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


def test_corrupt_config_is_backed_up_before_a_save_overwrites(isolated_home, capsys):
    bad = '{"port": 9000,}'  # trailing comma: invalid JSON
    (isolated_home / "config.json").write_text(bad)

    assert cfg_mod.load_config()["port"] == 8686  # still falls back...
    assert "unreadable" in capsys.readouterr().err  # ...but not silently

    # The bytes survive, so a later save cannot erase the file without a trace.
    backup = isolated_home / "config.json.corrupt"
    assert backup.read_text() == bad

    cfg = cfg_mod.load_config()
    cfg["port"] = 9999
    cfg_mod.save_config(cfg)
    assert json.loads((isolated_home / "config.json").read_text())["port"] == 9999
    assert backup.read_text() == bad


def test_config_that_is_not_an_object_is_backed_up(isolated_home):
    # Valid JSON of the wrong shape would otherwise crash on **data.
    (isolated_home / "config.json").write_text("[1, 2, 3]")

    assert cfg_mod.load_config()["port"] == 8686
    assert (isolated_home / "config.json.corrupt").read_text() == "[1, 2, 3]"


def test_coerce_resets_unknown_profile(isolated_home):
    (isolated_home / "config.json").write_text(json.dumps({"profile": "turbo"}))
    assert cfg_mod.load_config()["profile"] == "auto"


@pytest.mark.parametrize("profile", cfg_mod.profile_names())
def test_every_declared_profile_is_accepted(profile):
    assert cfg_mod._coerce_strict({"profile": profile})["profile"] == profile


def test_coerce_strict_rejects_unknown_profile():
    with pytest.raises(ValueError, match="profile"):
        cfg_mod._coerce_strict({"profile": "turbo"})


def test_default_backend_is_auto():
    # `auto` picks llama-server when its CUDA build sees the card and the
    # in-process engine otherwise, so neither is a config decision a user has
    # to make on hardware they have not measured.
    assert cfg_mod.DEFAULT_CONFIG["backend"] == "auto"


def test_coerce_resets_unknown_backend():
    assert cfg_mod._coerce({"backend": "torch"})["backend"] == "auto"


def test_parallel_has_to_be_at_least_one():
    assert cfg_mod._coerce({"parallel": 0})["parallel"] == 1
    with pytest.raises(ValueError, match="parallel"):
        cfg_mod._coerce_strict({"parallel": 0})


@pytest.mark.parametrize("backend", cfg_mod.backend_names())
def test_every_declared_backend_is_accepted(backend):
    assert cfg_mod._coerce_strict({"backend": backend})["backend"] == backend


def test_coerce_strict_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        cfg_mod._coerce_strict({"backend": "torch"})


def test_only_the_openai_layer_is_mounted_by_default():
    assert cfg_mod.DEFAULT_CONFIG["protocols"] == "openai"


def test_protocols_accept_a_comma_string(isolated_home):
    (isolated_home / "config.json").write_text(json.dumps({"protocols": "openai,ollama"}))
    assert cfg_mod.load_config()["protocols"] == "openai,ollama"


def test_protocols_accept_a_list_and_normalise_it(isolated_home):
    (isolated_home / "config.json").write_text(json.dumps({"protocols": ["Ollama", " ollama "]}))
    assert cfg_mod.load_config()["protocols"] == "ollama"


def test_unknown_protocol_warns_and_falls_back(isolated_home, capsys):
    (isolated_home / "config.json").write_text(json.dumps({"protocols": "mcp"}))

    assert cfg_mod.load_config()["protocols"] == "openai"
    assert "mcp" in capsys.readouterr().err


@pytest.mark.parametrize("protocol", cfg_mod.protocol_names())
def test_every_declared_protocol_is_accepted(protocol):
    assert cfg_mod._coerce_strict({"protocols": protocol})["protocols"] == protocol


def test_coerce_strict_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="protocols"):
        cfg_mod._coerce_strict({"protocols": "mcp"})


def test_speculative_ngram_is_off_by_default():
    # +37% on rewrite prompts, -1% / -2.1% on fresh prose: a default has to
    # hold for both, so the win stays opt-in.
    assert cfg_mod.DEFAULT_CONFIG["spec_type"] == "none"
