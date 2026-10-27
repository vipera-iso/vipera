from vipera.optimize.shaping import estimate_tokens, estimate_tool_tokens, shape

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


def test_dropping_history_never_orphans_a_tool_result():
    messages = [
        SYSTEM,
        {"role": "user", "content": "q0 " + "x" * 300},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "t0 " + "z" * 300},
        {"role": "assistant", "content": "a0 " + "y" * 300},
        {"role": "user", "content": "final"},
    ]

    for budget in range(80, estimate_tokens(messages)):
        out = shape(messages, budget_tokens=budget, max_tokens=0).messages
        pending_calls = False
        for message in out:
            if message.get("role") == "assistant":
                pending_calls = bool(message.get("tool_calls"))
            elif message.get("role") == "tool":
                assert pending_calls, f"orphaned tool result at budget {budget}: {out}"
            elif message.get("role") == "user":
                pending_calls = False


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


def test_a_tool_schema_is_charged_against_the_budget():
    # llama.cpp renders the schema into the prompt, so a request that fits as
    # messages alone can still overflow the context once tools are declared.
    messages = _history(6, chars=400)
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "description": "d" * 3000, "parameters": {}},
        }
    ]
    budget = estimate_tokens(messages) + 300

    assert shape(messages, budget_tokens=budget, max_tokens=0).applied is False

    out = shape(messages, budget_tokens=budget, max_tokens=0, tools=tools)

    assert out.applied is True
    assert estimate_tokens(out.messages) <= out.budget_tokens
    assert estimate_tokens(out.messages) + estimate_tool_tokens(tools) <= budget


def test_the_tool_schema_is_part_of_the_estimated_prompt():
    # /health's cache rate divides by these estimates, so the schema has to be
    # in them or a tool-heavy agent loop looks cheaper than it is.
    messages = _history(1)
    tools = [{"type": "function", "function": {"name": "read", "description": "d" * 600}}]

    out = shape(messages, budget_tokens=8192, max_tokens=128, tools=tools)

    assert out.applied is False
    assert out.estimated_before == estimate_tokens(messages) + estimate_tool_tokens(tools)
    assert out.estimated_after == out.estimated_before


def test_a_tool_schema_too_large_for_the_context_leaves_messages_alone():
    # Trimming messages cannot make the tools fit; the caller must see the
    # request unchanged rather than history silently deleted for nothing.
    messages = _history(4, chars=400)
    tools = [{"type": "function", "function": {"description": "d" * 6000}}]

    out = shape(messages, budget_tokens=2048, max_tokens=256, tools=tools)

    assert out.messages == messages
    assert out.applied is False


def test_shaping_never_loops_on_a_message_that_cannot_shrink():
    # Content is already shorter than the budget after the marker, so the third
    # stage has nothing left to cut and must stop.
    messages = [{"role": "user", "content": "hi"}]

    out = shape(messages, budget_tokens=10, max_tokens=0)

    assert out.messages[0]["content"] == "hi"
