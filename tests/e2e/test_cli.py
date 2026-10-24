"""Drive the installed CLI in a subprocess.

Unit tests call the Typer commands directly, which skips argument parsing,
console-script wiring and exit codes. These run `python -m vipera` so a broken
entry point, or a crash on an empty home, is caught before release.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import vipera

pytestmark = pytest.mark.e2e


@pytest.fixture
def run_cli(tmp_path):
    def _run(*args: str, home=None) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["VIPERA_HOME"] = str(home or tmp_path)
        # Windows runners default to cp1252; the CLI prints box-drawing glyphs.
        # The child writes UTF-8, so the parent must decode the pipes as UTF-8
        # too: with `text=True` alone the reader thread decodes cp1252, raises,
        # and leaves stdout/stderr as None.
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, "-m", "vipera", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=120,
        )

    return _run


def test_version_prints_the_package_version(run_cli):
    result = run_cli("--version")

    assert result.returncode == 0
    assert vipera.__version__ in result.stdout


def test_help_prints_the_usage_guide(run_cli):
    result = run_cli("help")

    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "vipera model search" in combined
    assert "vipera chat repl" in combined


def test_unknown_command_exits_nonzero(run_cli):
    result = run_cli("definitely-not-a-command")

    assert result.returncode != 0


def test_empty_home_exits_with_a_hint_not_a_traceback(run_cli):
    result = run_cli()

    combined = (result.stdout + result.stderr).lower()
    assert "Traceback" not in result.stderr
    assert "no models" in combined
    assert result.returncode == 1


def test_serve_status_reports_offline(run_cli, tmp_path):
    # Point the CLI at a port nothing listens on, so a developer's running
    # gateway on 8686 cannot turn this assertion into a false pass.
    (tmp_path / "config.json").write_text(json.dumps({"host": "127.0.0.1", "port": 1}))

    result = run_cli("serve", "status")

    assert result.returncode == 0
    assert "OFFLINE" in result.stdout.upper()
