import sys
import types

from vipera.commands import system_cmd


def test_doctor_prints_install_hints_when_runtime_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setitem(sys.modules, "gguf", None)

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "python -m pip install llama-cpp-python" in out
    assert "python -m pip install gguf" in out


def test_doctor_omits_install_hints_when_runtime_present(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))

    system_cmd.doctor()
    out = capsys.readouterr().out

    assert "pip install llama-cpp-python" not in out
    assert "pip install gguf" not in out
