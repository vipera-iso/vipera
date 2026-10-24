"""Tests that touch real files and real processes.

Nothing here mocks the filesystem or the network stack: the CLI writes an actual
config, and the HTTP check starts an actual gateway. Tests that need a real GGUF
skip unless `VIPERA_TEST_MODEL` points at one, so the suite stays runnable on a
machine with no models — CI sets it when a fixture model is available.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

TEST_MODEL = os.environ.get("VIPERA_TEST_MODEL")


def _cli(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["VIPERA_HOME"] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "vipera", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )


def test_config_written_by_the_cli_survives_a_reread(tmp_path):
    result = _cli(tmp_path, "system", "config", "port", "9011")
    assert result.returncode == 0, result.stderr

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["port"] == 9011

    # A second process must see the same value.
    reread = _cli(tmp_path, "system", "config", "port")
    assert reread.returncode == 0
    assert "9011" in reread.stdout


def test_cli_rejects_unknown_config_key_without_touching_the_file(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"port": 8686}))
    before = (tmp_path / "config.json").read_bytes()

    result = _cli(tmp_path, "system", "config", "no_such_key", "1")

    assert result.returncode != 0
    assert (tmp_path / "config.json").read_bytes() == before


def test_model_list_on_an_empty_home(tmp_path):
    result = _cli(tmp_path, "model", "list")

    assert result.returncode == 0
    assert "no models" in result.stdout.lower()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise AssertionError(f"gateway never became healthy at {base_url}")


@pytest.mark.skipif(not TEST_MODEL, reason="set VIPERA_TEST_MODEL to a .gguf to run")
def test_gateway_serves_a_real_model(tmp_path):
    port = _free_port()
    env = dict(os.environ)
    env["VIPERA_HOME"] = str(tmp_path)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vipera",
            "serve",
            "start",
            "--model",
            str(TEST_MODEL),
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base)

        models = httpx.get(f"{base}/v1/models", timeout=10).json()
        assert models["data"], "the gateway must advertise the loaded model"

        reply = httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            },
            timeout=120,
        )
        assert reply.status_code == 200
        assert reply.json()["choices"][0]["message"]["role"] == "assistant"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
