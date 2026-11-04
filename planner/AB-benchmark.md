# A/B: mistral.rs vs llama.cpp on the target machine — 2026-11-04

Gate: same command before/after, median of 3 runs, identical model files, same
context. Engines under comparison:

| Engine | Launch | Notes |
|---|---|---|
| llama.cpp | `vipera.cpp/build-cuda/bin/llama-server` (GGML_CUDA=ON) | the fork, reference |
| mistral.rs | `~/.cargo/bin/mistralrs-server --isq Q4K` (from HF, or GGUF load) | Rust engine |

Models, exact bytes shared by both engines:

- `~/.vipera/models/q8src-imbf-q3km.gguf` (7.3 GB, Q3_K_M)
- `~/.vipera/models/hermes4-14b-q4km.gguf` (8.4 GB, Q4_K_M)

Both sit fully in scarce VRAM on the 4 GB GTX 1650 Max-Q? No — neither fits:
GTX 1650 Max-Q = 4 GB, so `-ngl 24` (pre-measured P2 plan) layers move to GPU
and the rest stays in RAM. Same `--ngl` pinned for both engines. Measure:

  python3 tools/fb-bridge/bench_ab.py --engine llama-server --model <gguf> \
      --prompt 512 --gen 64
  python3 tools/fb-bridge/bench_ab.py --engine mistralrs   --model <gguf> \
      --prompt 512 --gen 64

pp/throughput from `/health` + `/completion` (llama-server) and `/v1/chat/completions`
(mistralrs `OpenAI`-shaped server). Two fixes reserved:

* P2.1: GGML_CUDA=ON llama-server build + MoE offload on
* P2.3: spec `ngram` cache (+37% decode measured on the fork)

Each config: median of 3 runs after 1 warm-up, per `PERF.md`.

---

## Head-to-head measurement — 2026-11-04 11:13 +07

- **Environment:** i5-11300H (8 threads), 15 GiB RAM + 16 GiB swapfile (zram 4 G), NVIDIA
  GTX 1650 (sm_75, 4096 MiB), CUDA toolkit **13.4** (`nvcc 13.4.92`). llama.cpp fork build
  **0.7.0-dev (build 18, commit 977cf94)**, GGML_CUDA.
- **Method:** **different domain — no head-to-head ratio is computed.** The two subjects do
  not perform the same job, so a "% faster than" figure between them would be meaningless
  and is deliberately *not* stated. Each side is reported in its own table with its own
  metric, per the anti-fabrication rule.
  - *Island Rust* = the vipera planner/scheduler hot path (`_zipf_sample`/`_routing` +
    `ExpertCache`), a **CPU-only** algorithm ported to PyO3. GPU utilisation 0 %.
  - *llama.cpp* = **LLM inference**, generating tokens on the GPU.

### A — Island Rust vs baseline Python (planner domain, CPU-only)

Workload: `simulate`, modelled routing, `glm-5.3-flash` on `hw/gtx1650.toml`, 512 tokens,
seed 0, context 2048. Identical arch/target/seed for both arms; the *only* variable is
`schedule._rs` (island module vs `None`). Source unmodified for both arms. 1 warm-up call
discarded, then 3 timed calls, **median**. Manifest sha256 identical across both arms.

| Config | Python baseline (median) | Island Rust (median) | Speedup | % faster |
|---|---|---|---|---|
| `quant=Quant.q4_experts()` (canonical workload of `tests/test_rs_island.py`) | 1680.6 ms `[1691.9, 1680.3, 1680.6]` | 465.8 ms `[464.8, 470.4, 465.8]` | **3.61×** | 72.3 % |
| ↳ repeat run (stability check) | 1633.8 ms `[1629.7, 1633.8, 1636.3]` | 456.5 ms `[456.5, 458.9, 454.9]` | **3.58×** | 72.1 % |
| `quant=Quant.fp8()` (`simulate` default) | 1553.2 ms `[1556.3, 1551.2, 1553.2]` | 376.4 ms `[376.1, 376.4, 379.0]` | **4.13×** | 75.8 % |

Resource record (planner side): CPU ≈ **99.5 %** of one core (single-threaded), peak RSS
≈ **27 MB**, GPU utilisation **0 %**, GPU 43–44 °C (idle).

### B — llama.cpp (inference domain, GPU)

Model `~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` (1 111 370 240 bytes,
`qwen2 1.5B Q4_K - Medium`), fully offloaded (`-ngl 99`), prefill 512 / gen 64, backends
CUDA, `n_threads 4`. 1 warm-up invocation discarded, then 3 timed invocations, **median**.

| Metric (median of 3) | Value | Runs |
|---|---|---|
| Prefill throughput | **440.445 t/s** | `[440.46, 440.39, 440.45]` |
| Decode throughput | **104.114 t/s** | `[104.17, 104.11, 103.94]` |

Per-run internal σ ≈ 0.12–0.19 t/s (≈0.18 %) — well inside run-to-run noise. Resource
record: GPU temp **53 → 54 → 55 °C** across the three runs, GPU utilisation high during
eval, model resident in VRAM.

### Conclusion

**INCOMPARABLE** — the two sides are in different domains (scheduling-algorithm
throughput vs LLM token generation). There is intentionally **no** "Rust is X % faster
than llama.cpp" line: the planner never generates a token and llama.cpp never runs the
scheduler. Each side has its own metric above.

### 3× verification (against `PERF.md`)

Measured island-vs-baseline-Python speedup = **3.61×** (canonical `q4_experts` workload)
and **4.13×** (`fp8` default), both median-of-3 with byte-identical manifests.

- The `3×` figure quoted in the previous commit is **not reproduced as stated**:
  measured **+20.3 %** (3.61×) and **+37.7 %** (4.13×) above 3× — both exceed the 20 %
  tolerance, so this is logged as a **mismatch (understated: actual ≈ 3.6–4.1×)**.
- `PERF.md`'s own island row states `4140 → 376 ms (11.0×)`. The **island arm reproduces
  exactly** (376.4 ms under `fp8`), but the **Python arm measures ≈1553 ms, not 4140 ms**,
  so the `11.0×` ratio was **not** reproduced either. `PERF.md` is left untouched per the
  no-silent-edit rule; this note is the only record of the discrepancy.

### Repro

```bash
# --- Island Rust arm: build from as-is source, then A/B the two arms ---
cd /home/user/vipera/planner/_rs && cargo build --release
mkdir -p /tmp/opencode/rs_probe
cp target/release/lib_planner_rs.so /tmp/opencode/rs_probe/_planner_rs.so
#   the harness imports vipera.schedule, toggles schedule._rs (module vs None),
#   calls simulate(arch, target, quant=..., context=2048, tokens=512, seed=0)
#   with 1 warm-up + 3 timed calls per arm, reports median; asserts manifest sha256 equal.
cd /home/user/vipera/planner && python3 <planner_a_b_harness>

# --- llama.cpp arm: 1 warm-up + 3 timed, median of avg_ts ---
/home/user/vipera.cpp/build-cuda/bin/llama-bench \
  -m /home/user/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -ngl 99 -p 512 -n 64 -o json        # discard run 0, median of runs 1-3
```

### Caveats

1. **No cross-domain ratio exists.** Any single "% faster" number spanning these two
   systems would be fabricated; only per-domain metrics are reported.
2. **The planner measurement is CPU-bound and single-core** (≈99.5 % of one thread, 27 MB
   RSS); it uses no GPU, so it is not competing with llama.cpp for VRAM or thermals.
3. **`PERF.md` numbers are partly unreproducible here** (Python arm 4140 ms / 11.0×): the
   island arm matches to the millisecond but the Python baseline measures ~1.55–1.68 s.
   The absolute times also depend on system load and the exact `quant`; both are recorded
   so the delta can be re-derived rather than trusted from `PERF.md` alone.
