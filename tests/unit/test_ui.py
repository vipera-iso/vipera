import io

import pytest
from rich.console import Console

from vipera import ui


@pytest.fixture
def captured(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(
        ui,
        "console",
        Console(file=stream, width=100, force_terminal=False, highlight=False),
    )
    return stream


def _flat(text):
    return " ".join(text.split())


def test_stream_shows_reasoning_and_returns_only_the_answer(captured):
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "thinking "}}]},
        {"choices": [{"delta": {"reasoning_content": "hard"}}]},
        {"choices": [{"delta": {"content": "Answer"}}]},
    ]

    text = ui._stream(iter(chunks), "assistant", refresh=30)

    assert text == "Answer"
    rendered = _flat(captured.getvalue())
    assert "thinking hard" in rendered
    assert "Answer" in rendered
    # Reasoning chunks are output: they count in the stats line.
    assert "3 tokens" in rendered


def test_stream_renders_a_reasoning_only_reply(captured):
    chunks = [{"choices": [{"delta": {"reasoning_content": "just thinking"}}]}]

    text = ui._stream(iter(chunks), "assistant", refresh=30)

    assert text == ""
    assert "just thinking" in _flat(captured.getvalue())
