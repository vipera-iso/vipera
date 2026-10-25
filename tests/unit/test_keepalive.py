import pytest

from vipera.keepalive import parse_keep_alive


def test_keep_alive_defaults_to_the_caller_value():
    assert parse_keep_alive(None, default=300) == 300
    assert parse_keep_alive(None, default=None) is None


def test_keep_alive_parses_ollama_durations():
    assert parse_keep_alive("5m") == 300
    assert parse_keep_alive("1.5h") == 5400
    assert parse_keep_alive("500ms") == 0.5
    assert parse_keep_alive("1d") == 86400
    assert parse_keep_alive(90) == 90
    assert parse_keep_alive("90") == 90


def test_keep_alive_zero_and_negative_are_special():
    assert parse_keep_alive("0") == 0
    assert parse_keep_alive(0) == 0
    assert parse_keep_alive(-1) == -1
    assert parse_keep_alive("-1") == -1
    assert parse_keep_alive("-2h") == -1


@pytest.mark.parametrize("value", ["soon", "5x", "", "  ", "m", "1.2.3m"])
def test_keep_alive_rejects_garbage(value):
    with pytest.raises(ValueError):
        parse_keep_alive(value)
