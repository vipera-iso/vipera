# Plan: replace one part at a time, by language, only where it pays

For every stage of both projects: what it costs now, what bounds it, which
implementation could replace it, what that replacement is worth **in ms or t/s on
this machine**, what it costs to own, and the verdict.

This is the language-specific companion to
`2026-11-05-physical-limit-program.md` (which decides *what* to optimize). Here
the question is narrower: **is a different implementation language the tool for
this stage?**

## The rule

A language replacement is adopted only when all four hold:

1. the stage is more than 1.2x its measured floor (otherwise there is nothing to
   win);
2. the stage is interpreter/CPU-bound -- not GPU, not disk, not a socket
   (a different language cannot make a memory-bound CUDA kernel faster);
3. the coarse-boundary test passes: the replacement is called once per plan,
   once per startup, or once per request -- never once per token (per-token FFI
   costs more than the work it saves);
4. the gain survives a re-measure after the replacement, and the build,
   packaging and CI cost is paid once, not per user.

A rejected stage is written down with its number, so it is not re-litigated from
memory (the CLI row below is exactly that case).

## Stage inventory, both projects

| # | stage | now | floor | candidate replacement | worth it? | verdict |
|---|---|---|---|---|---|---|
| 1 | process start (CLI) | 9.9 ms | 9.9 | none (native launcher saves the interpreter, not the work) | 0 | no |
| 2 | CLI imports + parse | 112.5 ms | ~10 | Rust/clap CLI | measured lazy ceiling 8 ms; a Rust CLI still spawns Python for the engine | no |
| 3 | config + registry | 0.1 ms | - | none | 0 | no |
| 4 | GGUF index parse | 2.9 ms | ~1 | **Rust island** (done, `gguf_rs`) | 4138 -> 2.9 ms | done |
| 5 | hardware probe | 10.9 ms warm / 27 ms cold | ~7 | Rust + `libnvidia-ml` dlopen instead of pynvml | ~14 ms cold | experiment E2 |
| 6 | planner (offline) | 376 ms | - | **Rust island** (done) | 4140 -> 376 ms | done |
| 7 | plan (live) | 0.01 ms | - | none | 0 | no |
| 8 | gateway relay | +1.30% wall | 0.8% | Rust/axum sidecar | <= 0.5% of wall | no |
| 9 | request JSON (in/out) | **0.46 ms/request** | ~0.4 (stdlib json) | `msgspec` (C) instead of pydantic models | 1.5-2.9% of gateway overhead | no (E1 done) |
| 10 | SSE relay per chunk | inside #8 | - | none | ~0 | no |
| 11 | tokenize / detokenize | llama.cpp C++ | - | already C++ | 0 | no |
| 12 | CPU quant mat-vec | 65 GB/s (part L3) | 45.4 GB/s DRAM | Rust/Zig CPU kernels | kernels already hit DRAM+L3; language changes nothing | no |
| 13 | GPU decode kernels | 116.6 GB/s | 188.6 GB/s | **no language** -- CUDA only | up to 1.6x t/s, by kernel technique | kernel track |
| 14 | GPU prefill | 440.6 t/s | 1015.7 t/s (FP16) | no language; dequant is the cost | unknown until F16 measured | kernel track |

## What each verdict rests on

**#2 CLI (no).** `serve_cmd` alone imports in 104.5 ms against 112.5 ms for the
whole CLI: only 8 ms belongs to the modules a given command does not use, and the
rest is typer (26 ms) + rich (16 ms) + the shared package. A Rust CLI would have
to reimplement five command surfaces and still launch Python for inference, so
the measured upside is 8 ms of a 150 ms startup -- inside the import's own noise.

**#4/#6 (done).** Both were the same shape: a per-item Python loop over data
(151k vocabulary strings, 4k planner samples). Both now run in Rust behind one
coarse call per operation, byte-identical output. This is what a successful
language replacement looks like in this codebase.

**#5 hardware probe (experiment).** 14 ms of the cold cost is `import pynvml`
(measured by `-X importtime`: 14.4 ms cumulative). A Rust function that
`dlopen`s `libnvidia-ml.so` and reads the same fields removes the import, at the
price of a second island to package. Small but real, and it reuses the `gguf_rs`
pattern: one call, plain return values, Python decides when to call it.

**#8 gateway (no).** Already gated: a Rust sidecar was measured at <= 1.3% against
a 0.8% passthrough floor. The remaining 0.5% of wall time is not what anyone
feels; the earlier +127% at c=4 turned out to be GPU serialization, not the
gateway, and was corrected by measuring `-np 4` directly.

**#9 JSON (measured, rejected).** `tools/json_cost.py` on a 20-message request
streaming 128 chunks: pydantic `model_validate_json` 0.019 ms, request
re-serialize 0.036 ms, chunk encode 0.0032 ms each, **0.46 ms per request** in
total (a stdlib `json.loads` of the same body is 0.013 ms, so pydantic's share
is ~0.006 ms). Against a gateway overhead of 31 ms that is 1.5%; against the
most pessimistic 16 ms it is 2.9%. The gate was 25%, so `msgspec` buys at most
0.4 ms per request -- below measurement noise and far below owning a second
serializer. No rewrite.

**#12 CPU kernels (no).** 65 GB/s against a 45.4 GB/s DRAM read is already above
DRAM -- the working set is partly in L3. A Rust or Zig kernel would have to beat
ggml's hand-tuned AVX2/AVX512 path while keeping every quant's exact numerics;
there is no measured gap to aim at, so there is nothing to replace.

**#13/#14 GPU (no language, yes technique).** The card reaches 188.6 GB/s on a
plain read, decode uses 116.6. The missing 1.6x lives inside CUDA kernels
(vector width, register blocking, fused dequant, `__dp4a`). Rust, Zig or Go
calling the same CUDA would produce the same kernel; Python would be a wrapper
with extra cost. Language is the wrong axis here.

## Experiments, in order

Each one: harness -> change -> re-measure -> keep only if it clears 5% and the
outputs stay identical. Stop the experiment when it fails twice.

### E1 -- how much is JSON? (done: rejected)

`tools/json_cost.py` (median-7): 0.46 ms of JSON per request, 1.5-2.9% of the
gateway's overhead against a 25% gate. Re-run with
`python3 tools/json_cost.py --gateway-overhead-ms 31` after any change to the
request or chunk models; it prints the share directly.

### E2 -- drop pynvml (vipera, new island function)

Add `_gguf_rs`-style `detect_gpu()` (or a small `hw_rs` module) returning
`(name, total_gb, free_gb)` from `libnvidia-ml`; keep pynvml as the fallback and
`VIPERA_HW_PURE=1` to force it. Gate: cold `vipera serve` startup at least 10 ms
faster, and `HardwareInfo` identical on this machine (same fields, same values
within 1%). Stop if the dlopen path is slower than the import.

### E3 -- kernel track (vipera.cpp, no language)

As in the physical-limit program: DRAM-resident kernel gate (~1 GB weight), then
find whether decode loss is in the mat-vec (compare t/s against bytes across two
quant sizes), then kernel changes only. Gate: >= 5% `llama-bench` decode t/s with
identical logits.

## Research status

`web_search` is returning unrelated pages for these queries (checked again while
writing this plan: the results were podcasts and package indexes, not
benchmarks), so **no external performance claim appears in this plan**. Every
number above is either measured here or arithmetic on a measurement.

When search works, three claims are worth checking before E1/E2 are funded:

1. `msgspec` vs pydantic v2 decode/encode throughput for the request shapes
   vipera serves;
2. published axum/tokio vs FastAPI overhead for streaming JSON responses;
3. how ik_llama.cpp organises its Q4_K dot (block layout, dp4a use) -- as a
   hypothesis to test with `kernel-gate`, never as a result to copy.

Any of those found in a blog post is a hypothesis. The gate decides, not the post.

## What this plan deliberately does not do

- No rewrite of the CLI, the gateway, or the HTTP layer in another language: each
  was measured, and each ceiling is smaller than the cost of owning a second
  implementation.
- No new dependency that is not already installed, except `msgspec` **if** E1
  proves the JSON layer is the bottleneck.
- No per-token FFI anywhere. The two islands that exist are both called once per
  operation, which is why they cost nothing at runtime.
