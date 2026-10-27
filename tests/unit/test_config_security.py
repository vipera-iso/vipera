"""Config plumbing for the hardening keys: normalizing, validating, permissions."""

import json
import stat

import pytest

from vipera import config


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the config module at a throwaway VIPERA_HOME."""
    monkeypatch.setattr(config, "VIPERA_HOME", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    return tmp_path


def test_the_keys_default_to_off():
    # Both features must be opt-in: an upgrading install keeps serving openly.
    assert config.DEFAULT_CONFIG["api_key"] is None
    assert config.DEFAULT_CONFIG["cors"] is None


def test_a_key_is_trimmed_and_an_empty_one_disables_the_check(home):
    (home / "config.json").write_text(json.dumps({"api_key": "  secret  "}))
    assert config.load_config()["api_key"] == "secret"

    (home / "config.json").write_text(json.dumps({"api_key": "   "}))
    assert config.load_config()["api_key"] is None


def test_an_invalid_key_is_refused_by_the_strict_writer(home):
    with pytest.raises(ValueError, match="api_key"):
        config._coerce_strict({"api_key": 123})


def test_cors_accepts_a_string_or_a_list(home):
    (home / "config.json").write_text(json.dumps({"cors": "http://a,http://b"}))
    assert config.load_config()["cors"] == "http://a,http://b"

    (home / "config.json").write_text(json.dumps({"cors": ["http://a"]}))
    assert config.load_config()["cors"] == ["http://a"]


def test_cors_rejects_a_shape_that_cannot_be_parsed(home):
    with pytest.raises(ValueError, match="cors"):
        config._coerce_strict({"cors": 42})


def test_the_config_file_is_not_world_readable(home):
    # It can hold an API key.
    config.save_config({**config.DEFAULT_CONFIG, "api_key": "secret"})

    mode = stat.S_IMODE((home / "config.json").stat().st_mode)
    assert mode & 0o077 == 0
