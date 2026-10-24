from vipera.optimize.prefix import PrefixStats, cached_tokens, prefix_key, reusable_prefix

SYSTEM = {"role": "system", "content": "you are a careful assistant."}
FITS = [
    SYSTEM,
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "an" * 10},
]


def test_the_prefix_stops_at_the_last_user_turn():
    messages = [*FITS, {"role": "user", "content": "next question"}]

    prefix = reusable_prefix(messages)

    assert prefix["messages"] == FITS


def test_the_current_turn_does_not_change_the_key():
    a = [*FITS, {"role": "user", "content": "question A"}]
    b = [*FITS, {"role": "user", "content": "a completely different question"}]

    assert prefix_key(a) == prefix_key(b)


def test_editing_the_system_prompt_changes_the_key():
    a = [SYSTEM, {"role": "user", "content": "hi"}]
    b = [{"role": "system", "content": "you are terse."}, {"role": "user", "content": "hi"}]

    assert prefix_key(a) != prefix_key(b)


def test_the_tool_schema_is_part_of_the_prefix():
    messages = [SYSTEM, {"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    assert prefix_key(messages, tools) != prefix_key(messages)


def test_a_request_with_nothing_shared_has_no_key():
    assert prefix_key([{"role": "user", "content": "hi"}]) == ""


def test_cached_tokens_reads_the_openai_shape():
    payload = {"usage": {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 768}}}

    assert cached_tokens(payload) == 768


def test_cached_tokens_falls_back_to_llama_server_timings():
    assert cached_tokens({"timings": {"cache_n": 512}}) == 512


def test_cached_tokens_is_zero_when_the_backend_does_not_say():
    assert cached_tokens({"usage": {"prompt_tokens": 10}}) == 0
    assert cached_tokens(None) == 0
    assert cached_tokens({"usage": {"prompt_tokens_details": {"cached_tokens": "n/a"}}}) == 0


def test_stats_count_a_repeat_prefix_as_a_hit():
    stats = PrefixStats()

    assert stats.record("abc", prompt_tokens=100, cached=0) is False
    assert stats.record("abc", prompt_tokens=100, cached=90) is True
    assert stats.record("zzz", prompt_tokens=50, cached=0) is False

    summary = stats.summary()
    assert summary["requests"] == 3
    assert summary["prefix_hits"] == 1
    assert summary["prefix_hit_rate"] == 0.333
    assert summary["prompt_tokens"] == 250
    assert summary["cached_tokens"] == 90
    assert summary["cache_rate"] == 0.36


def test_stats_ignore_an_empty_key_and_can_be_reset():
    stats = PrefixStats()

    stats.record("", prompt_tokens=10)
    assert stats.summary()["prefix_hits"] == 0

    stats.record("abc", prompt_tokens=10, dropped=2, truncated=1)
    assert stats.summary()["messages_dropped"] == 2
    assert stats.summary()["messages_truncated"] == 1

    stats.reset()
    assert stats.summary()["requests"] == 0
