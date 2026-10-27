"""Fit a chat request into the context the model was loaded with.

The planner picks ``context`` once, at load time. A client can still send more
history than that, and llama.cpp then either drops tokens itself (silently, from
the front, cutting the system prompt in half) or refuses the call. Shaping is
the layer in between: it keeps the messages that matter — the leading system
prompt and the current turn — and gives up older history first, then oversized
tool output, and only as a last resort part of the final message.

Token counts are estimated from characters (``_CHARS_PER_TOKEN`` is
deliberately pessimistic: English prose runs ~4 chars/token, code and JSON
~3). An estimate that is too high only trims a little early; an exact count
would need one tokenizer call per message, which on the `server` backend is one
HTTP round trip per message on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

# Rendered chat templates add a handful of control tokens per message.
_OVERHEAD_TOKENS = 4
# Conservative characters-per-token: over-count is safe, under-count is not.
_CHARS_PER_TOKEN = 3
# Leave room for the template's end-of-turn tokens and a little slack.
_RESERVE_TOKENS = 64
# When a message has to be cut, keep its head (instructions) and its tail
# (the actual question) and drop the middle.
_HEAD_SHARE = 0.75


@dataclass(frozen=True)
class Shaping:
    """What shaping did to a request. ``dropped``/``truncated`` are 0 when it fit."""

    messages: list[dict]
    dropped: int = 0
    truncated: int = 0
    estimated_before: int = 0
    estimated_after: int = 0
    budget_tokens: int = 0

    @property
    def applied(self) -> bool:
        return self.dropped > 0 or self.truncated > 0


def _marker(omitted: int) -> str:
    return f"\n\n[... {omitted} characters omitted ...]\n\n"


def _content(msg: dict) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _content_tokens(msg: dict) -> int:
    tokens = len(_content(msg)) // _CHARS_PER_TOKEN + _OVERHEAD_TOKENS
    # Tool calls are part of the rendered prompt too.
    if msg.get("tool_calls"):
        tokens += len(str(msg["tool_calls"])) // _CHARS_PER_TOKEN
    return tokens


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count for a whole request, template overhead included."""
    return sum(_content_tokens(m) for m in messages)


def estimate_tool_tokens(tools: list[dict] | None) -> int:
    """Tokens the tool definitions add to the prompt.

    llama.cpp renders every schema into the prompt before the messages, so a
    large tool set consumes the same context as history does. It cannot be
    trimmed (the model then stops calling the tool), so it has to reduce the
    room left for messages instead.
    """
    if not tools:
        return 0
    return len(str(tools)) // _CHARS_PER_TOKEN + _OVERHEAD_TOKENS


def _split(messages: list[dict]) -> tuple[int, int]:
    """Return (end of the protected system block, start of the current turn).

    Everything before ``head_end`` is a leading system prompt and everything
    from ``turn_start`` on is the turn being answered; both are kept.
    """
    head_end = 0
    while head_end < len(messages) and messages[head_end].get("role") == "system":
        head_end += 1

    turn_start = len(messages)
    for i in range(len(messages) - 1, head_end - 1, -1):
        if messages[i].get("role") == "user":
            turn_start = i
            break
    return head_end, turn_start


def _cut(text: str, keep_chars: int) -> str:
    """Keep the head and tail of ``text``, middle out."""
    if len(text) <= keep_chars:
        return text
    keep_chars = max(1, keep_chars)
    head = int(keep_chars * _HEAD_SHARE)
    tail = keep_chars - head
    omitted = len(text) - head - tail
    return text[:head] + _marker(omitted) + (text[-tail:] if tail > 0 else "")


def _shrink(msg: dict, excess_tokens: int) -> bool:
    """Cut ``msg`` enough to free ``excess_tokens``. True when it changed."""
    content = _content(msg)
    if not content:
        return False
    keep_chars = len(content) - max(1, excess_tokens) * _CHARS_PER_TOKEN
    if keep_chars >= len(content):
        return False
    # _cut keeps that many characters of content and then appends the marker.
    # Reserve the marker's length here, or the message stays a marker-length
    # over budget and the request never actually fits.
    marker = _marker(max(0, len(content) - keep_chars))
    target = max(1, keep_chars - len(marker))
    if target >= len(content):
        return False
    msg["content"] = _cut(content, target)
    return True


def shape(
    messages: list[dict],
    *,
    budget_tokens: int,
    max_tokens: int = 0,
    tools: list[dict] | None = None,
) -> Shaping:
    """Trim ``messages`` to ``budget_tokens``; return them untouched when they fit.

    ``budget_tokens`` is the model's context and ``max_tokens`` the room the
    answer needs, so the prompt budget is the difference (minus a small
    reserve). ``tools`` is the request's tool schema: it is charged against the
    same budget because the backend renders it into the prompt, and it is never
    trimmed. A non-positive budget means "no shaping" — the caller had no plan
    to read a context from.
    """
    if not messages or budget_tokens <= 0:
        return Shaping(messages=list(messages))

    tool_tokens = estimate_tool_tokens(tools)
    # The whole prompt budget, tools included: what estimated_before/after are
    # comparable to. Only the messages can be trimmed, so they get what is left.
    prompt_budget = budget_tokens - max(0, max_tokens) - _RESERVE_TOKENS
    message_budget = prompt_budget - tool_tokens
    if message_budget <= 0:
        # Even the tools do not fit; trimming messages cannot help.
        return Shaping(messages=list(messages))

    before = estimate_tokens(messages)
    if before <= message_budget:
        return Shaping(
            messages=list(messages),
            estimated_before=before + tool_tokens,
            estimated_after=before + tool_tokens,
            budget_tokens=prompt_budget,
        )

    head_end, turn_start = _split(messages)
    shaped = [dict(m) for m in messages]
    dropped = 0
    truncated = 0

    # 1. Cap any history message that takes more than a quarter of the budget.
    # A 200 KB tool result the client resent by accident is a blob, not
    # conversation: cutting it keeps the turn that refers to it answerable,
    # where dropping it whole would not.
    cap_tokens = max(64, message_budget // 4)
    for idx in sorted(
        range(head_end, turn_start), key=lambda i: _content_tokens(shaped[i]), reverse=True
    ):
        excess = _content_tokens(shaped[idx]) - cap_tokens
        if excess <= 0:
            break
        if _shrink(shaped[idx], excess):
            truncated += 1

    # 2. Still over: give up the oldest history, whole messages at a time. The
    # system block and the current turn stay: losing either changes the answer,
    # not just its length.
    while estimate_tokens(shaped) > message_budget and turn_start > head_end:
        # Drop a whole exchange: an assistant turn that called tools plus the
        # tool messages that answer it. Deleting them one at a time can leave a
        # `tool` message whose `assistant(tool_calls)` is gone, and the chat
        # template then rejects the request (or drops the result silently).
        end = head_end + 1
        while end < turn_start and shaped[end].get("role") == "tool":
            end += 1
        dropped += end - head_end
        del shaped[head_end:end]
        turn_start -= end - head_end

    # 3. Last resort: the current turn itself is too large (a pasted file). Keep
    # its head and tail so the instructions and the actual question survive, and
    # stop as soon as the request fits or nothing more can be removed.
    for _ in range(len(shaped)):
        excess = estimate_tokens(shaped) - message_budget
        if excess <= 0:
            break
        target = next(
            (i for i in range(len(shaped) - 1, -1, -1) if _content(shaped[i])),
            None,
        )
        if target is None or not _shrink(shaped[target], excess):
            break
        truncated += 1

    return Shaping(
        messages=shaped,
        dropped=dropped,
        truncated=truncated,
        estimated_before=before + tool_tokens,
        estimated_after=estimate_tokens(shaped) + tool_tokens,
        budget_tokens=prompt_budget,
    )
