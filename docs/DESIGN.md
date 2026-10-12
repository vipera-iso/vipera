# DESIGN — vipera: a model-file optimization layer for LLM/MoE on weak hardware

> System design. Positioning lives in [`OPTIMIZER.md`](OPTIMIZER.md) (the model-file optimization
> layer); deep algorithms live in [`ALGORITHMS.md`](ALGORITHMS.md). Bold numbers are measured on the
> machine or derived from a measurement.

## Context & decided assumptions

| Item | Decision | Source |
|---|---|---|
| Target device | **Personal computer** (i5-11300H, 15 GiB RAM, GTX 1650 4 GB, NVMe) | user choice + measurement |
| Target model | **Small MoE**: DeepSeek-V2-Lite (Phase 0/1), Qwen3-30B-A3B (stretch) | user choice |
| Budget | **0** — open source only, no cloud rental | user choice |
| Timeline | **1 month** | user choice |
| License | Open source, offline-first, cross-platform | original prompt |
| Mobile | **Later phase** (not in this month) | derived from "personal computer" |

**Stated assumption:** no Android device is available to test this month ⇒ anything "mobile" stays at
the design/portability level and is **not a measured result**. If that assumption is wrong, say so and
the scope changes.

---

## 2. System architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         API GATE                                  │
│  OpenAI-compatible /v1/chat/completions   ·   gRPC (later)        │
└───────────────▲──────────────────────────────────────────────────┘
                │ request
┌───────────────┴──────────────────────────────────────────────────┐
│                      RUNTIME ROUTER                               │
│  pick a runtime by (model, hardware):                             │
│    dense+GGUF → llama.cpp   ·  mobile → ExecuTorch / LiteRT       │
│    ONNX → ORT               ·  fallback → Python reference        │
└───────┬───────────────────────────────┬──────────────────────────┘
        │                               │
┌───────▼─────────┐           ┌─────────▼──────────────────────────┐
│ MODEL MANAGER   │           │      MEMORY MANAGER  ★ core ★       │
│ download/convert│           │  VRAM budget (4 GB)                 │
│ cache/quantize  │           │  ├─ attention + shared  (resident)  │
│ (GGUF, imatrix) │           │  ├─ expert cache (LRU/LFU) ★        │
└───────┬─────────┘           │  ├─ KV cache (paged, q8)            │
        │                     │  └─ host/disk spill (mmap, NVMe)    │
        │                     └─────────┬──────────────────────────┘
        │                               │
┌───────▼───────────────────────────────▼──────────────────────────┐
│                    EXECUTION ENGINE                               │
│  Expert scheduler: routing predictor → prefetch → compute         │
│  CPU + dGPU co-execution (layer-wise overlap)                     │
│  Feeds: llama.cpp (MoE CPU-offload) → consumes vipera manifest    │
└──────────────────────────────────────────────────────────────────┘
        ▲
        │  reads artifact + manifest produced by the OPTIMIZER (docs/OPTIMIZER.md)
┌───────┴──────────────────────────────────────────────────────────┐
│  vipera OPTIMIZER: analyze → transform (reorder/quant/prune) →    │
│  pack → model.vipera.gguf + model.vipera.json                     │
└──────────────────────────────────────────────────────────────────┘
```

**Components & status within one month:**

| Component | Role | In scope this month? |
|---|---|---|
| Memory Manager (expert cache + KV) | **core contribution**, differs from off-the-shelf | ✅ Phase 2 |
| Execution Engine (expert scheduler + prefetch) | predict experts, overlap I/O | ✅ Phase 2 |
| Optimizer (file-layer transforms + manifest) | **core positioning** | ✅ Phase 2b |
| Model Manager | download/convert GGUF, cache | ✅ Phase 0 |
| Runtime Router | pick a backend | ⚠️ only two branches (llama.cpp + Python ref) |
| API Gate | OpenAI-compatible | ✅ Phase 5 (minimal) |
| Mobile runtime | NNAPI/Core ML | ❌ roadmap |

---

## 3. Comparison with FreeToken and AirLLM

| Criterion | FreeToken | AirLLM | **vipera (proposed)** |
|---|---|---|---|
| Focus | MoE serving on consumer PCs | layer-wise streaming | **model-file optimization + MoE offload on weak hardware** |
| Offload unit | expert (fractional) | layer | expert + block |
| Strength | CPU–GPU co-exec, token-level split | runs a model larger than VRAM, simply | unifies both: stream when > RAM, cache when it fits |
| Limitation | assumes a capable GPU; not tuned for 4 GB | I/O-bound, slow | (new) |
| Published speedup | CPU–GPU 84–96% (T4), 11–89% (A10) | runs 671B on 12 GB | **target: ≥ 2× tok/s vs default llama.cpp CPU-offload on this machine** |
| Mobile-ready | partially | no | partially (backend is abstracted) |

**What vipera improves:** FreeToken optimizes for a *capable* consumer GPU; AirLLM optimizes for
*exceeding VRAM*. Both assume ample RAM. The target here (**4 GB VRAM + 15 GB RAM, no swap, DRAM-less
NVMe**) lands exactly where **both are weak**: the model fits RAM but overflows VRAM, and the disk is
slow. vipera's contribution is a **smart expert cache + routing prediction to cut I/O**, *plus* **a
model-file optimization layer** (expert reorder, residency manifest) so a weak runtime reads less —
not "fast streaming".

---

## 4. Roadmap (trimmed for one month / zero budget)

| Phase | Work | Measurable output | Week |
|---|---|---|---|
| **P0** | Baseline: build llama.cpp, run DeepSeek-V2-Lite Q4 (CPU-only & partial GPU), measure tok/s, RAM/VRAM | an honest baseline table | 1 |
| **P1** | Quant + KV: compare Q4_K_M/Q3_K_M, KV q8; measure quality (perplexity) vs speed | a trade-off table | 1–2 |
| **P2** | ★ **Expert cache (LRU/LFU) + routing prediction/prefetch** on MoE | cache hit %, tok/s vs P0 | 2–3 |
| **P2b** | ★ **File-layer optimizer**: reorder experts by frequency + residency manifest | disk bytes/token, cold-load time | 3 |
| **P3** | CPU + dGPU co-execution (overlap layers N/N+1) | tok/s vs P2 | 3 |
| **P4** | Paged KV + speculative (Qwen2.5-0.5B draft) | speedup | 4 (if time) |
| **P5** | OpenAI-compatible API + CLI + model manager | a working `curl` | 4 |

**Cut from this month (moved to roadmap v1.1+):** mobile runtimes (NNAPI/Core ML/ExecuTorch), ONNX
Runtime, gRPC, pruning/SVD/distillation, a full Qwen3-30B-A3B run.

**Minimum success criterion for the month:** run DeepSeek-V2-Lite on this machine, expose an
**OpenAI-compatible API**, and **prove with numbers** that the expert cache + prefetch clearly beats
baseline (idea: hit ≥ 60%, tok/s ≥ 1.5×). If it does not win, report honestly that it did not win.

---

## 5. Proposed tech stack (given zero budget)

| Layer | Choice | Reason |
|---|---|---|
| Core engine | **C++ (llama.cpp fork/patch)** | reuse existing kernels + GGUF + CUDA |
| Orchestration/cache | **Python 3** | fast experimentation, has `mmap`, `ctypes` |
| Custom kernels | C++ + CUDA (only if needed) | `nvcc` not installed — add later if required |
| Build | `cmake` + `gcc` (already present) | no cargo needed |
| Model format | **GGUF** (+ imatrix) | mature ecosystem, MoE offload support |
| Mobile (later) | Kotlin (NNAPI), Swift (Core ML), ExecuTorch | roadmap, not this month |
| Other runtimes | ONNX/MLC/ORT | reserved for the router, not built yet |

---

## 6. KPIs (measure your own baseline; borrow no numbers)

| Metric | How to measure |
|---|---|
| Tokens/s (prefill + decode) | count real tokens / wall-clock, ≥ 3 runs |
| Peak RSS | `/usr/bin/time -v` or `ps` sampling |
| Peak VRAM | `nvidia-smi --query-gpu=memory.used` |
| Expert cache hit % | instrument the cache, hit / total expert accesses |
| Disk I/O per token | `/proc/<pid>/io` read_bytes / tokens |
| First-token latency | timestamps before/after the first token |
| Perplexity | `llama-perplexity` on a wiki test set |
| Thermals/power | `nvidia-smi` + `powertop` (low priority on a PC) |

> **Do not** use "Recall@5" or FreeToken/AirLLM numbers as this project's KPIs — those are theirs.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **I/O bottleneck** (DRAM-less NVMe ~2 GB/s) | keep hot experts in VRAM/RAM; measure hit rate before optimizing reads |
| **Model exceeds RAM** (Qwen3-30B-A3B 18.6 GB > 11 GiB) | use DeepSeek-V2-Lite for P0–P2; Qwen3 only once streaming exists |
| **No swap** ⇒ OOM kills the process | use `--mlock` carefully, cap RSS, `systemd-run` limits |
| **Building llama.cpp with CUDA** may need `nvcc` | build CPU first (gcc/cmake present); CUDA only if needed |
| 404 GB free disk must hold several ~30 GB models | monitor free space |
| Quality loss from Q3/aggressive quantization | measure perplexity; do not drop bits if it rises > 0.3 |
| Model licensing | only use permissively licensed models (DeepSeek/Qwen allow it) |
| Laptop thermals/throttling | measure temperature; avoid > 30 min continuous runs |
| "Winning" not measurable ⇒ self-deception | fix the baseline criteria up front; report even when losing |
| No NPU/Android device to test | mark as "unverified", never claim it was done |

---

## 8. Next 7 days — action plan

**Day 1 — measurement foundation**
- [ ] Clone & build `llama.cpp` (CPU-only first) with `cmake`.
- [ ] Write `bench/` (Python): measure tok/s, RSS, VRAM, I/O per token, normalize output.
- [ ] Fix the baseline report format (table + JSON).

**Day 2 — P0 model**
- [ ] Download DeepSeek-V2-Lite GGUF `Q4_K_M` (~10 GB) and record checksums.
- [ ] Baseline runs: CPU-only (`-ngl 0`), partial (`-ngl 99 --n-cpu-moe`), record numbers.
- [ ] Confirm actual VRAM/RAM usage, fill in the table.

**Day 3 — quant & KV (P1)**
- [ ] Compare `Q4_K_M` vs `Q3_K_M`; KV `f16` vs `q8_0` vs `q4_0`.
- [ ] Measure perplexity for each configuration (short wiki test).
- [ ] Produce the quality/speed/memory trade-off table.

**Day 4 — expert cache design (P2)**
- [ ] Write the spec: cache key, LRU vs LFU vs hybrid policy, size based on free VRAM.
- [ ] Instrument llama.cpp/offload to **log which experts are selected** (data for prediction).
- [ ] Collect a routing trace of ≥ 1k tokens as a dataset.

**Day 5 — routing prediction (P2)**
- [ ] Analyze the trace: routing entropy, expert overlap between adjacent layers, potential hit rate.
- [ ] Prototype a simple predictor (frequency + last-used) and measure hit %.

**Day 6 — integrate & measure (P2)**
- [ ] Wire the cache + prefetch into the engine; rerun the Day-2 benchmark.
- [ ] Compare P0 vs P2: hit %, tok/s, I/O per token. Record it even if it loses.

**Day 7 — close the week & decide**
- [ ] Write `docs/RESULTS-W1.md` with the real numbers.
- [ ] Decide whether there is enough evidence for P3 (co-exec) or whether to return to cache tuning.
- [ ] Commit + tag `w1`.

---

## Flow of one decode step (P2 target)

```
token t
  │
  ├─ router layer n  ──► selects experts {e1,e2,...}
  │        │
  │        ├─ cache hit? ──► compute now (VRAM)
  │        └─ miss ──► load from RAM/disk (async) ──┐
  │                                                 │
  ├─ predictor: forecast layer n+1 experts ──► prefetch (hide latency)
  │
  └─ KV cache: allocate a page for the new token (no full KV copy)
```
