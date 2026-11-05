#!/usr/bin/env python3
"""What does the JSON layer cost per request? (experiment E1)

The gateway's own overhead was measured at 1.30% of wall time against a 0.8%
passthrough floor. Before replacing pydantic with `msgspec`, this answers how
many milliseconds of that overhead the JSON layer actually owns: validation of
the client body, re-serialization of the request for the backend, and one encode
per streamed chunk.

Usage:
    python3 tools/json_cost.py --chunks 128 --messages 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

# A realistic stream chunk: what openai.py builds per token.
CHUNK = {
    "id": "chatcmpl-8x1",
    "object": "chat.completion.chunk",
    "created": 1731000000,
    "model": "qwen2.5-1.5b-instruct-q4_k_m",
    "choices": [{"index": 0, "delta": {"content": " token"}, "finish_reason": None}],
}


def median_ms(fn, *, repeat: int = 7) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def build_body(messages: int, chars_per_message: int = 400) -> bytes:
    text = "lorem ipsum dolor sit amet " * (chars_per_message // 27 + 1)
    body = {
        "model": "qwen2.5-1.5b-instruct-q4_k_m",
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": text[:chars_per_message]}
            for i in range(messages)
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True,
    }
    return json.dumps(body).encode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=int, default=128)
    ap.add_argument("--messages", type=int, default=20)
    ap.add_argument(
        "--gateway-overhead-ms",
        type=float,
        default=0.0,
        help="absolute ms of gateway overhead for one request, from stage_gpu/gate wall",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from vipera.protocols.openai import ChatRequest

    body = build_body(args.messages)

    validate_ms = median_ms(lambda: ChatRequest.model_validate_json(body))
    req = ChatRequest.model_validate_json(body)
    dumps_request_ms = median_ms(
        lambda: json.dumps(
            {
                "model": req.model,
                "messages": [m.model_dump(exclude_none=True) for m in req.messages],
            }
        )
    )
    chunk_ms = median_ms(lambda: json.dumps(CHUNK, ensure_ascii=False))
    loads_ms = median_ms(lambda: json.loads(body))

    chunk_total = chunk_ms * args.chunks
    total_ms = validate_ms + dumps_request_ms + chunk_total
    share = (total_ms / args.gateway_overhead_ms * 100.0) if args.gateway_overhead_ms else None

    result = {
        "body_bytes": len(body),
        "messages": args.messages,
        "chunks": args.chunks,
        "validate_ms": round(validate_ms, 3),
        "dumps_request_ms": round(dumps_request_ms, 3),
        "per_chunk_ms": round(chunk_ms, 4),
        "chunks_total_ms": round(chunk_total, 3),
        "json_total_ms": round(total_ms, 3),
        "stdlib_json_loads_ms": round(loads_ms, 3),
        "gateway_overhead_ms": args.gateway_overhead_ms or None,
        "share_of_gateway_overhead_pct": None if share is None else round(share, 1),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"request body                 {result['body_bytes']:>8} bytes, {args.messages} messages")
    print(f"pydantic validate_json       {result['validate_ms']:>8.3f} ms")
    print(f"stdlib json.loads (floor)    {result['stdlib_json_loads_ms']:>8.3f} ms")
    print(f"request re-serialize         {result['dumps_request_ms']:>8.3f} ms")
    print(f"per stream chunk encode      {result['per_chunk_ms']:>8.4f} ms  x {args.chunks}")
    print(f"JSON total per request       {result['json_total_ms']:>8.3f} ms")
    if share is not None:
        print(f"share of gateway overhead    {share:>8.1f} %")
    print("\nGate to consider msgspec: JSON >= 25% of the gateway's own overhead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
