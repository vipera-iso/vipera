#!/usr/bin/env python3
"""Where do the seconds of `vipera serve` startup actually go?

Times each phase of the startup path in the same order `serve start` runs them,
then stops before uvicorn binds: config, model resolve (which parses the GGUF
index), hardware probe, plan, tuning replay, and engine load (spawn + health).
The last one is model load time; everything before it is Python work that a
different language could remove.

Usage:
    python3 tools/profile_startup.py --model qwen2.5-1.5b-instruct-q4_k_m
    python3 tools/profile_startup.py --model q8src-imbf-q3km --no-load
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import traceback

PHASES: list[tuple[str, float]] = []


def phase(name: str):
    class _P:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            PHASES.append((name, (time.perf_counter() - self.t0) * 1000.0))
            return False

    return _P()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-load", action="store_true", help="stop before spawning the engine")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for _ in range(args.repeat):
        PHASES.clear()
        engine = None
        try:
            with phase("import"):
                import vipera  # noqa: F401
                from vipera.backends.registry import create_engine, resolve_backend
                from vipera.config import load_config
                from vipera.hardware import detect
                from vipera.optimize.tuning import apply_tuning
                from vipera.planner import plan
                from vipera.registry import get_model

            with phase("config"):
                cfg = load_config()

            with phase("model resolve (gguf index)"):
                info = get_model(args.model)
                if info is None:
                    print(f"model not found: {args.model}")
                    return 1

            with phase("hardware probe"):
                hw = detect()

            with phase("plan"):
                p = plan(
                    info.meta,
                    hw,
                    profile="auto",
                    ctx=args.ctx,
                    kv_cache=cfg.get("kv_cache", "q8_0"),
                    kv_cache_k=cfg.get("kv_cache_k"),
                    kv_cache_v=cfg.get("kv_cache_v"),
                    no_kv_offload=cfg.get("no_kv_offload", False),
                    n_batch=cfg.get("n_batch", 512),
                    n_ubatch=cfg.get("n_ubatch", 0),
                    parallel=1,
                )

            with phase("tuning replay"):
                if cfg.get("tuning", True):
                    p = apply_tuning(p, info.alias, hw)

            with phase("resolve backend"):
                backend_used, p = resolve_backend(p, None)

            if not args.no_load:
                with phase("engine load (spawn + health)"):
                    engine = create_engine(p, info.path, backend_used)
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            if engine is not None:
                with contextlib.suppress(Exception):
                    engine.close()

    total = sum(ms for _, ms in PHASES)
    python_part = total - dict(PHASES).get("engine load (spawn + health)", 0.0)

    if args.json:
        print(json.dumps({"phases": PHASES, "total_ms": total, "python_ms": python_part}))
        return 0

    print(f"{'phase':<32} {'ms':>10} {'share':>8}")
    for name, ms in PHASES:
        share = (ms / total * 100.0) if total else 0.0
        print(f"{name:<32} {ms:>10.1f} {share:>7.1f}%")
    print(f"{'total':<32} {total:>10.1f}")
    print(f"{'of which Python (no engine load)':<32} {python_part:>10.1f}")
    if args.repeat > 1:
        print(f"(last of {args.repeat} runs; median would need per-phase capture)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
