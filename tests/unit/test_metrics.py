import time

from vipera.metrics import Meter, Stopwatch, Timings


def test_timings_split_a_request_into_prefill_and_decode():
    """The two numbers a user feels, from one stopwatch."""
    watch = Stopwatch()
    time.sleep(0.01)
    watch.first_token()
    time.sleep(0.02)
    watch.finish()

    timings = watch.timings({"prompt_tokens": 100, "completion_tokens": 20})

    assert 10 <= timings.prompt_ms <= 200
    assert 20 <= timings.decode_ms <= 200
    assert timings.prompt_tokens == 100
    # 20 tokens over a window of at least 20 ms: the rate has to be large.
    assert timings.decode_tps and timings.decode_tps > 50


def test_a_buffered_response_charges_everything_to_prefill():
    """No first token was ever observed, so there is no decode window to invent."""
    watch = Stopwatch()
    time.sleep(0.01)
    watch.finish()

    timings = watch.timings({"prompt_tokens": 5, "completion_tokens": 3})

    assert timings.decode_ms == 0
    assert timings.decode_tps is None
    assert timings.prompt_ms > 0


def test_timings_use_llama_cpp_key_names():
    payload = Timings(prompt_ms=500, decode_ms=1000, prompt_tokens=1000, completion_tokens=20)
    body = payload.to_dict()
    assert body["prompt_n"] == 1000
    assert body["predicted_n"] == 20
    assert body["prompt_per_second"] == 2000.0
    assert body["predicted_per_second"] == 20.0


def test_meter_accumulates_and_keeps_the_last_request():
    meter = Meter()
    meter.record(Timings(prompt_ms=100, decode_ms=900, prompt_tokens=100, completion_tokens=9))
    meter.record(Timings(prompt_ms=100, decode_ms=900, prompt_tokens=100, completion_tokens=9))

    snapshot = meter.snapshot()

    assert snapshot["requests"] == 2
    assert snapshot["prompt_tokens"] == 200
    assert snapshot["completion_tokens"] == 18
    assert snapshot["completion_tokens_per_second"] == 10.0
    assert snapshot["last"]["predicted_n"] == 9


def test_an_empty_meter_reports_no_rate_instead_of_zero():
    snapshot = Meter().snapshot()
    assert snapshot["requests"] == 0
    assert snapshot["completion_tokens_per_second"] is None
