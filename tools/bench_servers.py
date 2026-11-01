#!/usr/bin/env python3
"""Time an OpenAI-compatible server the way a user feels it.

Both Vipera's gateway and Ollama expose `/v1/chat/completions`, so one client
can time both with the same request, the same context length and the same
prompt - which is the only way a comparison between them means anything.

What is measured, all client-side so neither server grades its own homework:

* **TTFT** - request sent to the first content token. A prefill an order of
  magnitude slower than the other server shows up here, not in decode.
* **decode** - tokens per second from the first token to the last one, using the
  token count the server reports (`usage.completion_tokens`), falling back to
  counting deltas when it reports none.
* **total** - wall clock for the whole request.

The prompt is generated here, not read from a file, so two machines run the
same text: `--prompt-tokens` repetitions of a sentence that tokenises at about
17 tokens each. `--seed-prompt` adds a per-run marker, which defeats prompt
caching and measures a cold prefill every time; leave it off to measure what a
warm gateway does for a repeat request.

Usage:
    python tools/bench_servers.py --url http://127.0.0.1:8686/v1 --model qwen2.5-7b-q4km
    python tools/bench_servers.py --url http://127.0.0.1:11434/v1 --model vipera-bench-7b

Ollama's own `eval_count`/`eval_duration` can be printed as a cross-check with
`--ollama-native`, which reads `/api/generate` after the timed runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# One sentence, ~17 tokens on a Qwen tokenizer. Repeating it builds a prompt of
# a known approximate size without shipping a fixture nobody can verify.
_SENTENCE = (
    "The gateway measures the machine before it loads anything, budgets video "
    "memory per layer, and prints the plan it is about to run. "
)


def build_prompt(tokens: int, marker: str = "") -> str:
    """A deterministic prompt of roughly ``tokens`` tokens."""
    repeats = max(1, tokens // 17)
    text = _SENTENCE * repeats
    if marker:
        text = f"[run {marker}]\n" + text
    # Ask for a long answer on purpose: a model that stops after ten tokens gives
    # a decode rate dominated by fixed overhead, and the comparison stops being
    # about throughput. 200 tokens is enough window for the rate to settle.
    return text + "\n\nWrite a 150-word explanation of why running models locally matters."


def timed_request(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Stream one completion and return its timings."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()

    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first: float | None = None
    last = started
    deltas = 0
    usage_tokens: int | None = None
    prompt_tokens: int | None = None
    text: list[str] = []

    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens") or usage_tokens
                prompt_tokens = chunk["usage"].get("prompt_tokens") or prompt_tokens

            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if not piece:
                    continue
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                deltas += 1
                text.append(piece)

    total = last - started
    ttft = (first - started) if first is not None else float("nan")
    tokens = usage_tokens or deltas
    # Decode rate covers the window after the first token, so prefill does not
    # get billed at decode speed. With one token there is no window to divide.
    window = last - first if first is not None else 0.0
    decode_tps = (tokens - 1) / window if tokens > 1 and window > 0 else float("nan")

    return {
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "total_s": total,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "text": "".join(text).strip(),
    }


def ollama_native(url: str, model: str, prompt: str, max_tokens: int) -> dict | None:
    """Ollama's own counters, for a cross-check of the client-side numbers.

    `/api/generate` is Ollama-only, so this is never used for the comparison
    itself - only to confirm the client measurement lands in the same place.
    """
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}}
    ).encode()
    request = urllib.request.Request(
        f"{base}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError):
        return None
    eval_count = payload.get("eval_count") or 0
    eval_ns = payload.get("eval_duration") or 0
    prompt_ns = payload.get("prompt_eval_duration") or 0
    return {
        "decode_tps": (eval_count / (eval_ns / 1e9)) if eval_ns else float("nan"),
        "ttft_s": (prompt_ns / 1e9) if prompt_ns else float("nan"),
        "tokens": eval_count,
    }


def summarise(runs: list[dict]) -> dict:
    def median(key: str) -> float:
        values = [r[key] for r in runs if r[key] == r[key]]  # drop NaN
        return statistics.median(values) if values else float("nan")

    return {
        "ttft_s": median("ttft_s"),
        "decode_tps": median("decode_tps"),
        "total_s": median("total_s"),
        "tokens": runs[-1]["tokens"],
        "prompt_tokens": runs[-1]["prompt_tokens"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. :8686/v1")
    parser.add_argument("--model", required=True, help="Model name the server expects")
    parser.add_argument("--label", default="", help="Name to print for this server")
    parser.add_argument("--prompt-tokens", type=int, default=1000, help="Approximate prompt size")
    parser.add_argument("--max-tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs, after one warmup")
    parser.add_argument(
        "--seed-prompt",
        action="store_true",
        help="Add a per-run marker so prompt caching cannot hide prefill cost",
    )
    parser.add_argument(
        "--ollama-native", action="store_true", help="Also print Ollama's own counters"
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    args = parser.parse_args(argv)

    label = args.label or args.url

    def run(index: int) -> dict:
        marker = f"{time.time():.3f}" if args.seed_prompt else ""
        return timed_request(
            args.url, args.model, build_prompt(args.prompt_tokens, marker), args.max_tokens
        )

    try:
        warmup = run(0)
    except urllib.error.URLError as exc:
        print(f"{label}: cannot reach {args.url}: {exc}", file=sys.stderr)
        return 2

    runs = [run(i + 1) for i in range(args.runs)]
    summary = summarise(runs)
    summary["label"] = label
    summary["warmup_s"] = warmup["total_s"]

    if args.json:
        print(json.dumps(summary))
        return 0

    print(f"{label}")
    print(f"  prompt tokens   {summary['prompt_tokens'] or '?'}")
    print(f"  tokens out      {summary['tokens']}")
    print(f"  ttft            {summary['ttft_s'] * 1000:.0f} ms")
    print(f"  decode          {summary['decode_tps']:.2f} tok/s")
    print(f"  total           {summary['total_s']:.2f} s")

    if args.ollama_native:
        native = ollama_native(
            args.url, args.model, build_prompt(args.prompt_tokens), args.max_tokens
        )
        if native:
            print(
                f"  ollama native   ttft {native['ttft_s'] * 1000:.0f} ms, "
                f"decode {native['decode_tps']:.2f} tok/s ({native['tokens']} tokens)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
