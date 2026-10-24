from vipera.optimize.shaping import estimate_tokens, shape

SYSTEM = {"role": "system", "content": "you are a careful assistant."}


def _turn(i: int, chars: int = 300) -> list[dict]:
    return [
        {"role": "user", "content": f"question {i} " + "x" * chars},
        {"role": "assistant", "content": f"answer {i} " + "y" * chars},
    ]


def _history(turns: int, chars: int = 300) -> list[dict]:
    messages = [SYSTEM]
    for i in range(turns):
        messages.extend(_turn(i, chars))
    messages.append({"role": "user", "content": "now answer this"})
    return messages


def test_a_request_that_fits_is_returned_untouched():
    messages = _history(2)

    out = shape(messages, budget_tokens=8192, max_tokens=256)

    assert out.applied is False
    assert out.messages == messages
    assert out.messages is not messages  # copied, so callers cannot mutate ours


def test_no_budget_means_no_shaping():
    messages = _history(20, chars=2000)

    out = shape(messages, budget_tokens=0, max_tokens=256)

    assert out.applied is False
    assert out.messages == messages


def test_oldest_history_is_dropped_before_anything_is_truncated():
    messages = _history(10, chars=400)
    budget = estimate_tokens(messages) // 2

    out = shape(messages, budget_tokens=budget, max_tokens=0)

    assert out.dropped > 0
    assert out.truncated == 0
    # The system prompt and the current turn survive, the oldest turn does not.
    assert out.messages[0] == SYSTEM
    assert out.messages[-1]["content"] == "now answer this"
    assert "question 0 " not in "".join(m["content"] for m in out.messages)
    assert estimate_tokens(out.messages) <= out.budget_tokens


def test_a_huge_tool_result_is_truncated_in_the_middle_not_dropped():
    messages = [
        SYSTEM,
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "HEAD " + "z" * 20_000 + " TAIL"},
        {"role": "user", "content": "summarize it"},
    ]

    out = shape(messages, budget_tokens=2048, max_tokens=256)

    assert out.applied is True
    assert out.dropped == 0
    tool = out.messages[3]
    assert tool["role"] == "tool"
    assert len(tool["content"]) < 20_000
    assert tool["content"].startswith("HEAD")
    assert "characters omitted" in tool["content"]
    assert estimate_tokens(out.messages) <= out.budget_tokens


def test_the_final_user_message_is_cut_last_and_keeps_both_ends():
    messages = [
        SYSTEM,
        {"role": "user", "content": "BEGIN-INSTRUCTION " + "q" * 8000 + " END-QUESTION"},
    ]

    out = shape(messages, budget_tokens=1024, max_tokens=128)

    assert out.dropped == 0
    assert out.truncated >= 1
    content = out.messages[-1]["content"]
    assert content.startswith("BEGIN-INSTRUCTION")
    assert content.endswith("END-QUESTION")
    assert "characters omitted" in content


def test_multiple_leading_system_messages_are_all_protected():
    messages = [
        {"role": "system", "content": "first rule"},
        {"role": "system", "content": "second rule"},
        *_history(8, chars=400)[1:],
    ]

    out = shape(messages, budget_tokens=estimate_tokens(messages) // 3, max_tokens=0)

    assert out.messages[0]["content"] == "first rule"
    assert out.messages[1]["content"] == "second rule"


def test_shaping_never_loops_on_a_message_that_cannot_shrink():
    # Content is already shorter than the budget after the marker, so the third
    # stage has nothing left to cut and must stop.
    messages = [{"role": "user", "content": "hi"}]

    out = shape(messages, budget_tokens=10, max_tokens=0)

    assert out.messages[0]["content"] == "hi"
