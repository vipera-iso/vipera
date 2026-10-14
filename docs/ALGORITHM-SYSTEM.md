# ALGORITHM-SYSTEM — the executable half of the literature pack, and what it is worth

> `docs/reference/literature/` is a folder of papers. A folder of papers is not a
> system. This document covers the part that runs: four algorithms with a function
> behind each name, one harness that measures them against real artifacts, and the
> ledger of which proposed levers were taken and which were rejected — each with the
> number that decided it.
>
> **Status labels:** `[measured]` = reproducible, command quoted · `[simulated]` = a real
> algorithm over a constructed input · `[modelled]` = computed, not run · `[published]` =
> someone else's, not re-verified. A number without a label is a defect.
>
> Run it: `python3 -m vipera algo list` · `python3 -m vipera algo run <name>` ·
> `python3 -m bench.algo_measure` (writes `bench/results/algo-measure.json`).

---

## 1. The measurement that decides what any of this can be worth

No optimization in this repository can be judged without the tier of the problem it
attacks. On the target machine, for `DeepSeek-V2-Lite.Q4_K_M.gguf` at the default
invocation, 59.0 ms/token decomposes as:

| component | ms/token | share | label |
|---|---|---|---|
| ALU / compute | 43.7 | **74%** | `[measured]` |
| self-inflicted `mmap` paging | 15.3 | 26% | `[measured]` |
| real NVMe during decode | **≤0.1** | **≤0.2%** | `[measured]` |

The ALU figure sits at the modelled ceiling of 43.5 ms — **~0.5% headroom**. Decode
therefore has an *arithmetic* ceiling of ~23 tok/s on this machine, and the real file
measures 17.4–20.1 tok/s, i.e. the top of that range is 87% of the ceiling.

**The consequence is a rule every entry below obeys:**

> A lever can only pay if it changes **FLOPs per token**. A lever that changes
> **bytes** can only pay at a granularity the runtime can execute, in a regime the
> machine can reach, and this machine reaches neither.

The single cleanest proof is that the manifest cut disk reads by **26%** and
throughput moved by **−4%** `[measured]`. Bytes are 0.2% of the problem; a 26% cut to
0.2% is 0.05%.

---

## 2. The registry

Four entries, each a paper mapped to a function that can be executed and checked.
`kind` groups them by the resource they act on.

| name | kind | source | status | what it proved |
|---|---|---|---|---|
| `kv-retention` | kv | 2309.17453 (StreamingLLM), 2306.14048 (H2O) | `[simulated]` | H2O wins the objective it was written for (retained attention mass) and **loses** the per-step one; the two rankings disagree, and printing only the first would be scoring H2O against itself |
| `quant-gptq` | quant | 2210.17323 (GPTQ) | `[measured]` | on **real** Q4_K weights, Hessian norm **2.09× better** while Frobenius **1.54× worse** — two objectives, two numbers |
| `select-submodular` | select | krause-golovin14 | `[measured]` | on the **real** 511-token OLMoE trace the objective is **modular**, so greedy ties the exact optimum and the (1−1/e) bound is **vacuous** there |
| `spec-draft-length` | spec | 2302.01318 / 2211.17192 | `[exact]` | closed form for expected accepted length cross-checked against a Monte-Carlo of the actual accept/reject/residual sampler |

The status column is a claim about **where the input came from**, not that an entry is
fully measured. `quant-gptq` is `[measured]` because its *weights* are real; the
Hessian it needs comes from no capture that exists in this repository, so it stays
synthetic and the summary says so.

---

## 3. What each one proved on a real artifact

`bench/algo_measure.py`, output in `bench/results/algo-measure.json`.

### 3.1 `quant-gptq` — real Q4_K weights, a verified dequantiser

The harness reads `blk.0.ffn_gate.weight` out of `models/Qwen3-0.6B-Q4_K_M.gguf`
directly: 48 Q4_K super-blocks, 6912 bytes, 12288 weights. No `llama-quantize`
binary exists in this checkout, so the Q4_K layout was transcribed from
`ggml/src/ggml-quants.c` and then **verified, not trusted** — every value is
re-encoded and must reproduce the nibble it came from, which a wrong sub-block scale,
a swapped scale/min field or a wrong nibble half all break:

```
dequantiser round-trip   100.0000%   (12288 values, worst error 0.00e+00)
```

The decode is exactly invertible, so the numbers below are about GPTQ and not about
the reader. Both quantisers share one bit grid taken from the untouched weights, so a
difference in error is a difference in **diffusion**, never in bit allocation:

| | round-to-nearest | GPTQ | ratio |
|---|---|---|---|
| Frobenius MSE | 4.943e-5 | 7.624e-5 | **1.54× worse** |
| Hessian objective | 0.012450 | 0.005952 | **2.09× better** |

That is the paper's claim, in its own terms, on this project's data: GPTQ lowers the
norm that tracks the layer's output error and *pays* in the norm round-to-nearest
minimizes. The control arm is in the run: with `H = I` the inverse is diagonal, nothing
propagates, and GPTQ must reduce **bit-for-bit** to round-to-nearest — which it does
(`identity_control_identical: true`), so the propagation is acting on the Hessian
rather than on its own.

### 3.2 `select-submodular` — the branch went vacuous, and that is the result

On `bench/results/olmoe-1b-7b-511tok.jsonl` (511 captured decode steps, 16 layers, 64
experts, OLMoE's own top-8 routing; the *selection budget* under test is k = 6, which
retains 26622 of the 65408 selections):

```
prefix == greedy on 16/16 layers;  min ratio to exact optimum 1.0
submodular_bound_applies: false
```

A trace defines a **modular** objective: keeping expert `e` serves its own selection
count and nothing overlaps, so `f(S) = Σ_{e∈S} freq(e)` and the optimum is simply the
top-k by frequency — the rule `vipera/prune.py` already used. Greedy, lazy greedy and
the frequency prefix return the same set on every layer, and the exact optimum
computed by sorting confirms zero loss.

The (1−1/e) guarantee therefore bounds a ratio that is 1.0. Saying so is more useful
than presenting the bound as if it were doing work. The submodular machinery earns its
place on the synthetic covering instance, where the objective is genuinely
non-modular — and on that instance it is the `[exact]` theorem that the tests assert.

### 3.3 The other two

`kv-retention` runs over a synthetic attention stream, so it is `[simulated]` and its
number describes the algorithm, not the machine. It is still worth having, because it
is the arm that shows the two reasonable objectives for KV eviction **disagree**.
`spec-draft-length` is `[exact]`: the closed form and a 20000-trial Monte-Carlo agree,
which is what makes the formula trustworthy before it is ever used on a real draft
model.

---

## 4. The ledger — taken, and rejected

Every lever the literature pack proposed, and the number that decided it. This is the
document's main purpose: a rejection with a number is a result, and a proposal that is
neither taken nor rejected is just a PDF.

### Taken

| lever | number | label |
|---|---|---|
| `-t <physical cores> -lm mlock` | 12.80 → **22.90 tok/s = 1.79×** | `[measured]` |
| FlashAttention (`-fa on`) | **+4.6%** (22.46→23.49) · **+4.0%** (14.96→15.56), p=0.0079, 4/4 A/B pairs | `[measured]` |
| StreamingLLM sink + sliding window, `--keep 32` | **4.1× more tokens** before stopping, price **+0.64% PPL** (inside the noise) | `[measured]` |
| GPTQ error diffusion (§3.1) | Hessian norm 2.09× better | `[measured]` weights, `[simulated]` Hessian |
| Expert cache `lfu` over `hot` on *stationary* real routing | 0.97× vs 0.93× of the matched ceiling | `[measured]` on the OLMoE trace |

### Rejected

| lever | the number that killed it | label |
|---|---|---|
| the runtime manifest (vipera's own core) | **0.92–1.00×** on a clean baseline; the 1.22× is **retracted** | `[measured]` |
| quantized KV cache | **1.000×**, p=0.68 — the two configurations are the same measurement to within the instrument | `[measured]` |
| GPU offload (CUDA / Vulkan) | **0.43×** / **0.21×** of an unaided CPU | `[measured]` |
| activation sparsity | **0.0%** dead channels in a shipped gated FFN — there is nothing to skip | `[measured]` |
| retention by accumulated surprise | loses to recency at an equal budget: **31.93 vs 21.45 PPL** (5.8–19σ) | `[measured]` |
| speculative decoding | **not testable**: the build loads the draft and reports `speculative.types: none`; the 0.991× is the cost of an unused draft, not a result | `[measured]` |
| rewriting expert slabs | round-trip is **377/377 lossless**, but PPL **4.54 → 671.79** because `ffn_gate_inp` keeps routing by the old order | `[measured]` |
| expert pruning at the modelled 5% sparsity | only 21–25% of routing mass survives, so **53–71% of experts** must be kept for 95% of the mass | `[modelled]` |
| per-layer cache allocation | does not help | `[measured]` |

### The pattern in the rejection column

Every rejection acts on **bytes or placement**: disk traffic, cache bytes, KV width,
weight slab layout, device placement. Every acceptance that runs at the file layer is
worth ≈0, and the two acceptances that moved the measured number (+1.79×, +4%) are
**runtime flags**, not file transforms.

**vipera's own layer measured ~0%.** The `README` already says this under *"What
vipera claims now, and what it no longer claims"*; this ledger is the evidence table
behind that sentence.

---

## 5. What this implies — the three levers that are left

A model is `FLOPs/token` against an ALU ceiling. Only three things move that:

1. **Fewer active parameters per token.** The machine has 0.93 GiB of non-expert
   weights in a 30B model and 16.35 GiB of experts `[measured]`. Dropping experts
   *without* breaking quality means moving `ffn_gate_inp` with the slabs — which is
   the difference between the 4.54 → 671.79 failure recorded above and a real win.
2. **Fewer forward passes per token.** That is speculative decoding, and it is
   blocked on a build where the lever exists, not on a model question.
3. **Doing the arithmetic somewhere, or in some way, the ceiling does not tax.**
   Kernel and thread configuration is the empirical form of this — it is where the
   only 1.79× came from.

Everything else — placement, cache policy, quantized KV, disk layout — is bounded by
the 0.2% row in §1 and by what the runtime refuses to execute. The manifest expresses
per-expert VRAM placement, per-layer KV placement and per-expert disk streaming; the
runtime offers `-ngl` (whole layers), `-ctk/-ctv` (all the KV) and **no per-expert
streaming at all**. On CPU-only the manifest degrades to KV quantization, which §4
measures at 1.000×.

---

## 6. What is not claimed

- **Nothing here generalises past the machine.** The FLOPs verdict is a statement
  about a 4C/8T laptop running a 2.4B-active MoE that nearly fills RAM. A machine
  where the model fits in VRAM is a different problem with different degrees of
  freedom — and there, byte-level levers would face a different (bandwidth) ceiling.
- **The spill regime has never been measured.** It was shown to be *unreachable* on
  this machine (a model large enough to spill is a model too large to load: exit 137
  during load, twice). That is a statement about this machine's RAM, not about
  whether placement pays when it is possible.
- **The GPTQ Hessian is synthetic.** No activation capture exists in this repository.
- **The GPTQ comparison is at layer scale, not model scale.** `dense_inverse` is
  O(cols³) in pure Python; this module is a reference comparison, not a quantizer for
  a 10 GB file. The 2.09× is what the algorithm does, and it is not a tok/s claim.
- **The `select-submodular` bound is vacuous on the trace.** The `[exact]` status
  refers to the synthetic instance where the objective is non-modular.
- **The submodular/KV/GPTQ entries change no measured tok/s.** They are `[exact]` or
  `[simulated]` properties; the two entries in §4's *Taken* table that carry a tok/s
  number are `-t/-lm mlock` and FlashAttention, and neither of those is an algorithm
  in this registry.
