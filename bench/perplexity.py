"""Run llama-perplexity and record a quality number next to the speed numbers (S10).

`bench/bench.py` measures how *fast* a model runs; this measures how *good* it is.
vipera's scheduler does not model quality at all, so this is the other axis: it
establishes a baseline, and it is the harness any compression/pruning experiment
has to be judged with.

    python3 -m bench.perplexity \\
        --bin ~/dev/llama.cpp/build/bin/llama-perplexity \\
        --model models/olmoe-q4km.gguf \\
        --file ~/dev/llama.cpp/wikitext-2-raw/wiki.test.raw \\
        --chunks 20 --json bench/results/olmoe-ppl.json

Anchoring the corpus matters: perplexity is only comparable across runs that use
the same text, context and chunk count. The canonical choice is wikitext-2 test
(`llama.cpp/scripts/get-wikitext-2.sh`); use the whole set for a publishable
number, and a fixed `--chunks` for a quick regression.

Long-context quality (`--compare-stream`)
-----------------------------------------
`vipera run --stream` keeps `--keep` attention sinks and slides the KV window
(docs/REWRITE.md, stage B). That buys unbounded generation, and the bill is paid
in quality: the middle of the context is dropped. This harness measures that bill
by scoring the *same* tokens twice - once with the plan's full window, once with
sink + sliding attention - and reporting both PPLs and the gap.

Both runs must use the same `--context`, `--stride` and `--chunks`; the streaming
side is a different attention pattern, not a different evaluation, so the two
numbers are directly comparable. The gap is reported next to the standard error
of the difference, and the report prints the gap **in sigmas** rather than only
a verdict: `significant` is a deliberate 1-sigma screen ("worth looking at"), so
`bench/perplexity.py` says that in its own output and never implies a 95% test.
Read the sigma: ~2 is the usual bar, and 195 chunks of wikitext-2 is what it
takes to separate the sink counts in docs/REWRITE.md §B.

The stock (non-streaming) strided path clears its KV cache every window; the
streaming path never does. Emulating it requires a one-line dispatch added to
`llama.cpp/tools/perplexity/perplexity.cpp`: with `--context-shift` plus a
positive `--keep`, and `--ppl-stride > 0`, the strided evaluation keeps the KV
cache alive and shifts it exactly like `llama-cli` does. Without that patch
`--keep`/`--context-shift` are accepted and silently ignored by the tool.

`--stride` is what selects the strided evaluation at all (`-c` alone runs the
per-chunk evaluation, which resets its cache and cannot express streaming), and
it forces `-b` to the context size: the tool otherwise derives `n_parallel` from
`n_batch/n_ctx` and then fails to find a memory slot for a full-size batch.

Exit codes: 0 measured, 1 the run failed or no PPL was printed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

DEFAULT_BIN = "llama-perplexity"

# "Final estimate: PPL = 8.6426 +/- 0.30373"
PPL_RE = re.compile(r"PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
# "llama_perplexity: 13 context shifts, 5121 tokens scored, sinks = 4"
SCORED_RE = re.compile(r"([0-9]+)\s+tokens scored")
SHIFTS_RE = re.compile(r"([0-9]+)\s+context shifts")


def parse_ppl(text: str) -> tuple[float, float] | None:
    matches = PPL_RE.findall(text)
    if not matches:
        return None
    value, stderr = matches[-1]
    return float(value), float(stderr)


def parse_stream_stats(text: str) -> tuple[int | None, int | None]:
    """(tokens scored, context shifts) from the streaming path, or (None, None)."""
    scored = SCORED_RE.findall(text)
    shifts = SHIFTS_RE.findall(text)
    return (
        int(scored[-1]) if scored else None,
        int(shifts[-1]) if shifts else None,
    )


def build_command(binary: str, model: str, *, corpus: str, chunks: int | None = None,
                  context: int = 512, threads: int | None = None, ngl: int | None = 0,
                  stride: int | None = None, keep: int | None = None,
                  context_shift: bool = False) -> list[str]:
    """The llama-perplexity invocation for one configuration.

    `stride` turns on the strided (windowed) evaluation and pins `-b` to the
    context size; `keep` + `context_shift` switch that evaluation to sink +
    sliding attention. Both are needed for the streaming side.
    """
    cmd = [binary, "-m", model, "-f", corpus, "-c", str(context)]
    if stride is not None:
        # the tool derives n_parallel = n_batch/n_ctx and then cannot place a
        # full-size batch in that many sequences - keep batch == context.
        cmd += ["-b", str(context), "--ppl-stride", str(stride)]
    if keep is not None:
        cmd += ["--keep", str(keep)]
    if context_shift:
        cmd += ["--context-shift"]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    return cmd


#: KV eviction policies the patched llama-perplexity understands. The policy is not a
#: command-line flag: the patch reads it from the environment, which is the same
#: env-driven rule the rest of vipera's llama.cpp patches follow.
EVICTION_POLICIES = ("recency", "recency-gap", "surprise", "surprise-local")
POLICY_ENV = "LLAMA_PPL_STREAM_POLICY"
#: `surprise-local` protects this many of the newest tokens (env-driven, like the policy).
MARGIN_ENV = "LLAMA_PPL_STREAM_MARGIN"
DEFAULT_MARGIN = 128


def run_perplexity(binary: str, model: str, *, corpus: str, chunks: int | None = None,
                   context: int = 512, threads: int | None = None, ngl: int | None = 0,
                   timeout: float | None = None, stride: int | None = None,
                   keep: int | None = None, context_shift: bool = False,
                   policy: str | None = None, margin: int | None = None) -> dict:
    cmd = build_command(binary, model, corpus=corpus, chunks=chunks, context=context,
                        threads=threads, ngl=ngl, stride=stride, keep=keep,
                        context_shift=context_shift)

    env = None
    if policy is not None:
        if policy not in EVICTION_POLICIES:
            raise ValueError(f"unknown eviction policy {policy!r}; expected one of {EVICTION_POLICIES}")
        env = {**os.environ, POLICY_ENV: policy}
    if margin is not None:
        env = {**(env or os.environ), MARGIN_ENV: str(margin)}

    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "command": cmd,
                "elapsed_s": round(time.monotonic() - started, 3)}
    elapsed = time.monotonic() - started

    output = proc.stdout + proc.stderr
    report = {"ok": False, "exit_code": proc.returncode, "command": cmd,
              "elapsed_s": round(elapsed, 3), "chunks": chunks,
              "context": context, "corpus": os.path.basename(corpus),
              "stride": stride, "keep": keep, "context_shift": context_shift,
              "policy": policy, "margin": margin}
    scored, shifts = parse_stream_stats(output)
    if scored is not None:
        report["scored_tokens"] = scored
    if shifts is not None:
        report["context_shifts"] = shifts
    ppl = parse_ppl(output)
    if ppl is None:
        report["error"] = "no PPL line in output"
        report["stderr_tail"] = proc.stderr[-2000:]
        return report
    report["ok"] = proc.returncode == 0
    report["ppl"], report["ppl_stderr"] = ppl
    return report


def compare_stream(binary: str, model: str, *, corpus: str, chunks: int | None,
                   context: int, stride: int, keep: int, threads: int | None = None,
                   ngl: int | None = 0, timeout: float | None = None,
                   policy: str | None = None) -> dict:
    """Score the same tokens with the full window and with sink + sliding attention."""
    stock = run_perplexity(binary, model, corpus=corpus, chunks=chunks, context=context,
                           threads=threads, ngl=ngl, timeout=timeout, stride=stride)
    stream = run_perplexity(binary, model, corpus=corpus, chunks=chunks, context=context,
                            threads=threads, ngl=ngl, timeout=timeout, stride=stride,
                            keep=keep, context_shift=True, policy=policy)

    report = {"ok": bool(stock.get("ok") and stream.get("ok")),
              "context": context, "stride": stride, "keep": keep, "chunks": chunks,
              "corpus": os.path.basename(corpus), "policy": policy,
              "stock": stock, "stream": stream}
    if not report["ok"]:
        return report

    report["ppl_stock"] = stock["ppl"]
    report["ppl_stream"] = stream["ppl"]
    report["delta_ppl"] = stream["ppl"] - stock["ppl"]
    report["delta_pct"] = 100.0 * (stream["ppl"] - stock["ppl"]) / stock["ppl"]
    # Is the gap larger than the noise on both sides? Two independent estimates,
    # so the standard error of the difference is the quadrature sum.
    se = (stock["ppl_stderr"] ** 2 + stream["ppl_stderr"] ** 2) ** 0.5
    report["difference_stderr"] = se
    # `significant` is a 1-sigma *screen*, not a 95% test: it says "worth looking
    # at", and `sigmas` is reported alongside so the strength is never implied.
    # A 95% two-sided claim needs 2 sigma (docs/REWRITE.md §B).
    report["difference_sigmas"] = abs(report["delta_ppl"]) / se if se > 0 else 0.0
    report["significant"] = report["difference_sigmas"] > 1.0
    return report


def format_table(report: dict) -> str:
    lines = [f"ok            {report.get('ok')}",
             f"exit          {report.get('exit_code')}",
             f"elapsed       {report.get('elapsed_s')} s"]
    if report.get("ppl") is not None:
        lines.append(f"corpus        {report['corpus']}  ({report['chunks']} chunks, "
                     f"ctx {report['context']})")
        if report.get("keep") is not None:
            lines.append(f"attention     sink + sliding  (keep {report['keep']}, window "
                         f"{report['context'] - report['keep']}, "
                         f"policy {report.get('policy') or 'recency'})")
        if report.get("scored_tokens") is not None:
            lines.append(f"scored        {report['scored_tokens']} tokens, "
                         f"{report.get('context_shifts', 0)} context shifts")
        lines.append(f"PPL           {report['ppl']:.4f} +/- {report['ppl_stderr']:.5f}")
    if report.get("error"):
        lines.append(f"error         {report['error']}")
    return "\n".join(lines)


def difference_sigmas(report: dict) -> float:
    """How many standard errors the gap is, derived when the report predates the field.

    Reports are meant to be kept and re-read (docs/REWRITE.md §B quotes them), so a
    stored comparison from an older revision must still render instead of raising on
    a key it never wrote.
    """
    sigmas = report.get("difference_sigmas")
    if sigmas is not None:
        return float(sigmas)
    se = report.get("difference_stderr") or 0.0
    return abs(report.get("delta_ppl", 0.0)) / se if se > 0 else 0.0


def format_comparison(report: dict) -> str:
    lines = [f"corpus        {report['corpus']}  (ctx {report['context']}, "
             f"stride {report['stride']}, chunks {report['chunks']})",
             f"sinks         {report['keep']}"]
    stock, stream = report.get("stock", {}), report.get("stream", {})
    if stock.get("ppl") is not None:
        lines.append(f"full window   PPL {stock['ppl']:.4f} +/- {stock['ppl_stderr']:.5f}"
                     f"  ({stock['elapsed_s']} s)")
    else:
        lines.append(f"full window   FAILED: {stock.get('error', stock.get('exit_code'))}")
    if stream.get("ppl") is not None:
        lines.append(f"sink+sliding  PPL {stream['ppl']:.4f} +/- {stream['ppl_stderr']:.5f}"
                     f"  ({stream['elapsed_s']} s, {stream.get('context_shifts', 0)} shifts)")
    else:
        lines.append(f"sink+sliding  FAILED: {stream.get('error', stream.get('exit_code'))}")
    if report.get("ok"):
        lines.append(f"delta         {report['delta_ppl']:+.4f} PPL "
                     f"({report['delta_pct']:+.2f}%)")
        lines.append(f"noise         +/- {report['difference_stderr']:.4f} PPL "
                     f"(standard error of the difference)")
        sigmas = difference_sigmas(report)
        lines.append(f"strength      {sigmas:.1f} sigma "
                     f"(this is a 1-sigma screen; a 95% claim needs 2)")
        lines.append(f"verdict       "
                     f"{'gap is larger than the noise' if sigmas > 1.0 else 'gap is within the noise'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="perplexity", description=__doc__)
    p.add_argument("--bin", default=DEFAULT_BIN)
    p.add_argument("--model", required=True)
    p.add_argument("--file", required=True, dest="corpus", help="text corpus")
    p.add_argument("--chunks", type=int, default=None,
                   help="limit chunks (omit for the full set)")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--stride", type=int, default=None,
                   help="ppl-stride; enables the strided (windowed) evaluation")
    p.add_argument("--keep", type=int, default=None,
                   help="attention sinks to keep (streaming side of --compare-stream)")
    p.add_argument("--context-shift", action="store_true",
                   help="slide the KV window instead of stopping when it is full")
    p.add_argument("--policy", choices=list(EVICTION_POLICIES), default=None,
                   help="which tokens the sliding window evicts (via LLAMA_PPL_STREAM_POLICY)")
    p.add_argument("--margin", type=int, default=None,
                   help=f"tokens the `surprise-local` policy always keeps (default {DEFAULT_MARGIN})")
    p.add_argument("--compare-stream", action="store_true",
                   help="run the full window and sink + sliding on the same tokens")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--ngl", type=int, default=0)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--json", dest="json_path")
    args = p.parse_args(argv)

    binary = shutil.which(args.bin) or args.bin

    if args.compare_stream:
        if args.stride is None or args.keep is None:
            print("--compare-stream needs --stride and --keep", file=sys.stderr)
            return 2
        report = compare_stream(binary, args.model, corpus=args.corpus, chunks=args.chunks,
                                context=args.context, stride=args.stride, keep=args.keep,
                                threads=args.threads, ngl=args.ngl, timeout=args.timeout,
                                policy=args.policy)
        print(format_comparison(report))
    else:
        report = run_perplexity(binary, args.model, corpus=args.corpus, chunks=args.chunks,
                                context=args.context, threads=args.threads, ngl=args.ngl,
                                timeout=args.timeout, stride=args.stride, keep=args.keep,
                                context_shift=args.context_shift, policy=args.policy,
                                margin=args.margin)
        print(format_table(report))

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"report written to {args.json_path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
