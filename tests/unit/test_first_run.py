"""The no-argument entry point: a fresh install has to end in a gateway."""

from types import SimpleNamespace

from typer.testing import CliRunner

from vipera import cli
from vipera.commands import model_cmd, serve_cmd


def _wire(monkeypatch, fetched, started):
    specs = [SimpleNamespace(alias="starter")]

    monkeypatch.setattr(cli, "list_models", lambda: specs if fetched else [])
    monkeypatch.setattr(
        cli, "load_config", lambda: {"host": "localhost", "default_model": "starter"}
    )
    monkeypatch.setattr(cli, "print_banner", lambda **kwargs: None)
    monkeypatch.setattr(model_cmd, "fetch_starter", lambda: fetched.append(True))
    monkeypatch.setattr(serve_cmd, "start", lambda **kwargs: started.append(kwargs))


def test_first_run_fetches_a_starter_model_then_starts(monkeypatch):
    fetched, started = [], []
    _wire(monkeypatch, fetched, started)
    monkeypatch.delenv("VIPERA_AUTO_FETCH", raising=False)

    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 0, result.output
    assert fetched == [True]
    assert started == [{"model": "starter"}]


def test_auto_fetch_can_be_turned_off(monkeypatch):
    fetched, started = [], []
    _wire(monkeypatch, fetched, started)
    monkeypatch.setenv("VIPERA_AUTO_FETCH", "0")

    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 1
    assert fetched == []
    assert started == []
