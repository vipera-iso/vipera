# Program: drive every stage of both projects to its physical floor

Two projects, one question per stage: **how far is it from the wall, and what
moves it?** A stage at 90% of its roofline is finished no matter how ugly the
code is; a stage at 5% is a target even if the code is beautiful.

Scope of this document: the stage inventory, the measured baseline, the floor
for each stage, and the order to work in. It is not a promise of speedups -- every
number here was produced by running something on this machine (GTX 1650 4 GiB,
i5-11300H, 15 GiB RAM), and every future change is kept only if it clears noise
on a re-run.

## Measurement harness

| tool | project | what it answers |
|---|---|---|
| `vipera/tools/pipeline_audit.py` | vipera | cost and roofline headroom per Python/CPU stage, plus the GPU stages |
| `vipera/tools/stage_gpu.json` | vipera | the recorded llama-bench baseline the audit reads (rewrite by re-running, never by hand) |
| `vipera/tools/loop_survey.py` | vipera | per-item cost of every Python loop that scales with input size |
| `vipera/tools/profile_startup.py` | vipera | phase-by-phase `serve` startup |
| `vipera.cpp/tools/kernel_gate` + `compare.py` | vipera.cpp | quant kernel GB/s against the memory-bandwidth ceiling |
| `llama-bench` | vipera.cpp | end-to-end prefill/decode t/s for the same model |

Rule for the whole program: one change at a time, retest with the same harness,
keep only a result that clears the noise band (>= 5% and outside the observed
spread), and record the raw JSON next to the harness.

## Baseline, measured

### vipera -- CPU stages

| stage | measured | floor | headroom | verdict |
|---|---|---|---|---|
| process start (bare python) | 9.3 ms | 9.3 ms | 1.0x | done |
| CLI: import + parse | 145.8 ms | ~9 ms | 15.8x | **target** |
| config load | 0.05 ms | - | - | done |
| registry resolve | 0.08 ms | - | - | done |
| GGUF index parse | 2.9 ms | ~1 ms | ~3x | done (was 4138 ms) |
| hardware probe | 10.9 ms | ~7 ms | 1.6x | driver bound |
| plan | 0.01 ms | - | - | done |

The CLI line is the only CPU stage with room. What is in it (median-5, separate
processes): bare interpreter 9.9 ms, `import vipera.cli` 112.5 ms, click group
build 3.9 ms more. Of the 112.5, only 8 ms is the four command modules the
invoked command does not need: `serve_cmd` alone imports in 104.5 ms, so typer
26 ms, rich 16 ms, the backends tree and the rest are shared by every command.

**A lazy command registry is therefore not worth it**: its ceiling is 8 ms
(5% of startup, and inside the 103-112 ms run-to-run spread of the import alone),
paid for by restructuring typer registration, the root `--help` still having to
import everything, and every test that drives `CliRunner(vipera.cli.app)`.
What did pay was the eager `vipera.sources`/`psutil` chain (commit `ba6cebc`),
which was 28 ms of that shared tree and is now gone.

### vipera -- GPU stages (llama-bench, `-ngl 99`, Qwen2.5-1.5B Q4_K_M)

| stage | measured | roofline | headroom | bound by |
|---|---|---|---|---|
| decode | 104.4 t/s | 171.8 t/s (192 GB/s / 1.12 GB) | 0.61x | weight bandwidth |
| prefill (512) | 440.6 t/s | 1015.7 t/s (3.61 TFLOPS FP16 / 2*1.78 GFLOP) | 0.43x | compute |
| gateway overhead | 1.30% | 0.8% passthrough | 1.62x | socket relay |

### vipera.cpp -- CPU mat-vec (`kernel-gate`, 4096x4096, batch 1, 4 threads)

| quant | GB/s | % of the card's 192 GB/s |
|---|---|---|
| Q8_0 | 52.6 | 27% |
| Q4_K | 65.0 | 34% |
| Q3_K | 41.6 | 22% |

This is the CPU path (`ggml-cpu`), which is what a MoE offload or a CPU-only box
runs -- it is not the CUDA kernel the GPU decode uses. Read it as "ggml's CPU
mat-vec reaches a third of DRAM", not as "the GPU kernel does".

Caveat on the CPU row: the 4096x4096 Q4_K weight is 9.4 MB and the i5-11300H has
8 MB of L3, so part of that 65 GB/s is cache, not DRAM (the CPU read probe below
measures 45.5 GB/s from DRAM). A DRAM-resident gate needs a ~1 GB weight.

### Hardware ceiling, measured (`bandwidth-probe`)

| probe | GB/s | vs spec |
|---|---|---|
| GPU read (1 GiB, `bandwidth-probe-cuda`) | **188.6** | 98% of 192 |
| GPU copy (read+write) | 172.8 | 90% |
| CPU read (1 GiB, 4 threads) | **45.5** | - |
| CPU copy (read+write) | 39.2 | - |

The GPU reaches essentially its spec bandwidth with a plain read kernel, so the
spec sheet is a fair ceiling there -- which makes decode's 116.6 GB/s of traffic
62% of the achievable 188.6, not an artefact of an inflated headline number.

## What the numbers say

1. **Decode is 62% of the measured bandwidth ceiling, and that gap is real.**
   The probe reads 188.6 GB/s from this card, so 104.4 t/s over 1.117 GB of
   weights (116.6 GB/s) leaves ~1.6x on the table -- worth up to ~169 t/s if the
   decode chain were a pure stream. It is not a pure stream (attention, KV reads
   and activations all touch memory too), so the honest target is the part of
   that 1.6x the kernels own. The CPU numbers say the same thing one level down:
   ggml's CPU mat-vec reaches 65 GB/s against 45.5 GB/s of DRAM read (the
   remainder is L3), which is what the MoE offload and CPU fallback pay.

2. **A different language does not help the kernels.** The gap, whatever its
   size, is bandwidth efficiency inside a kernel: register blocking, vector
   loads, whether the dequant is fused with the dot. Rust, Zig, or Go calling
   the same CUDA would produce the same kernel; rewriting in Python would be
   slower. The language replacement question is closed for both the GPU and the
   CPU kernel side.

3. **Prefill at 43% of compute roofline is not a kernel bug.** K-quants dequant
   to FP16 on the fly; a GTX 1650 has no tensor cores for that path, so the
   roofline number assumes unpacks that the hardware does not have. Measure a
   dequant-free (F16) prefill before treating this as headroom.

4. **The gateway is within 1.6x of a pure socket relay, and its floor is not 0%.**
   Nothing language-level to win here; a Rust sidecar was already measured at
   <= 1.3% in the earlier gate.

5. **The Python side is nearly done.** GGUF parse went 4138 -> 2.9 ms, planner
   4140 -> 376 ms, per-item loops are all under 6 ms. What remains is process
   startup, which is a product decision (how much CLI UX to pay for), not an
   algorithmic one.

## Order of work

Ranked by (gap to floor) x (how much of a request it touches):

1. **Close the kernel gap (vipera.cpp).** The wall is measured: 188.6 GB/s, and
   decode uses 116.6. Next experiments, one at a time, each judged by the same
   probe plus `llama-bench`: (a) grow `kernel-gate`'s weight to ~1 GB so the CPU
   numbers stop being cache-resident; (b) find which part of a decode step is
   off the roofline by timing the same model at two quant sizes and checking
   whether t/s scales as 1/bytes -- if it does, the loss is not in the mat-vec;
   (c) only then touch kernel code (vector width, `__dp4a`, fused dequant).
   Gate: >= 5% on `llama-bench` decode, identical logits on a fixed prompt.
2. **CLI startup (vipera) -- closed, not worth it.** Measured ceiling 8 ms for a
   lazy command registry (see above), under both the 5% rule and the noise band.
   The remaining ~90 ms is typer + rich + the shared package, i.e. the price of
   the CLI's own UX, and cutting it is a product decision, not an optimization.
3. **Prefill dequant path.** Only after 1: measure F16 prefill on the same card
   to see whether the 43% is dequant or a real compute wall. If dequant, it is
   the same fix as 1.
4. **Model load time.** Not yet measured end to end; 1.12 GB from disk plus the
   VRAM copy. Measure before touching.

## Research status

`web_search` is returning empty results on this machine right now, so the
external comparison (ik_llama.cpp kernel design, published bandwidth numbers)
could not be done in this pass. Nothing above depends on it: every claim is a
local measurement or arithmetic on one. Re-run the search before adopting an
outside kernel, and treat any outside claim as a hypothesis to test with
`kernel-gate` rather than as a result.

## Stopping condition

A stage stops when a change fails to clear 5% twice, or when it is within 1.2x of
its floor. The program ends when every stage is in one of those two states --
that is the physical limit reachable from this hardware and this code, and it is
a statement about this machine, not about the algorithm in general.
