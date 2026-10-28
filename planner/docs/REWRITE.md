# REWRITE — where a real advantage can actually come from

> Goal: make vipera **win on the target machine**, not just describe it. The repo's own
> measurements say most of what it tried cannot win, so this file records the levers in
> order of whether they can move a measured number, and what each one costs. It is a
> status log: **A, B and C are implemented and measured; D was decided against on a measured
> probe, and E (retention by surprise) was built, measured and lost to recency.**

Status labels are the repo's own: `[measured]` reproducible with a quoted command ·
`[modelled]` computed, not run · `[proposed]` not implemented.

---

## 0. The constraint every stage has to face

`plantodo.md` STEP 1 and `docs/FINDINGS.md`, both `[measured]`:

- decode on the target laptop is **ALU-bound**: `mlock` reaches 43.7 ms/token against a
  modelled **43.5 ms** ALU floor (~0.5% headroom, ~23 tok/s ceiling), and the ALU is a
  measured 110 GFLOP/s (`bench/hwprobe.c`);
- real NVMe traffic during decode is **≤0.2%** of the time;
- the default `mmap` path wastes **26%** of decode re-faulting evicted pages.

⇒ A lever can only win by (a) removing the self-inflicted paging tax, (b) **cutting
FLOPs/token** (the only thing that raises the ALU ceiling), or (c) creating a *capability*
the measured footing does not have (long context). Saving disk bytes on a RAM-resident
model was measured to buy **nothing to −4%** (`docs/FINDINGS.md` §Round 4).

---

## A. Launch policy — **DONE, `[measured]`**

`vipera/launch.py` + wiring in `vipera/run.py` (`RunFlags.load_mode`, `build_command`) and
`vipera/cli.py` (`--load-mode`, automatic `-t`).

It chooses the llama.cpp load mode from the model size and the machine, so the process
stops paying the 26% paging tax:

| condition | mode | why |
|---|---|---|
| fits RAM and memlock limit covers it | `mlock` | pin; no page can be evicted mid-decode |
| fits RAM but memlock is shorter | `dio` | bypass the page cache |
| does not fit RAM | `mmap` | paging is unavoidable |

Threads default to the **physical** core count (ALU-bound workload ⇒ hyperthreads contend).

**Measured on `models/DeepSeek-V2-Lite.Q4_K_M.gguf`**, ctx 512, 32 tokens, CPU-only build,
`vipera run … --target hw/gtx1650.toml --compare`, three sequential runs (never concurrent —
see `docs/FINDINGS.md` §2):

| run | baseline (`mmap`) | manifest (`dio -t 4`) | speedup |
|---|---|---|---|
| 1 | 13.50 tok/s | 20.40 | **1.51×** |
| 2 | 13.10 | 22.80 | **1.74×** |
| 3 | 10.70 | 22.70 | **2.12×** |

The manifest side is stable (~20–23 tok/s, ≈ the ALU ceiling); the baseline is noisy because
`mmap` thrashes. Reported as a range, never at its favourable end: **~1.5–2.1×, median 1.74×**.
Peak RSS −1219 MiB and disk read −5549 MiB in the first run.

**Why this is a real advantage and not a flag trick:** the measurement is the product's own
claim, on the machine, with the command quoted. What it is *not* is a new algorithm — it is
vipera finally owning the one lever its own data proved. Nothing here touches the ALU
ceiling, so it cannot go beyond ~23 tok/s; that ceiling is stage D's problem.

Tests: `tests/test_launch.py` (policy branches, helpers, `-lm` mapping). Suite: **304 passed**
(`pytest` and `unittest`, same count either way).

---

## B. Long context — **DONE (budget + policy + wiring), `[measured]`**

`vipera/longcontext.py` turns `arch.kv_bytes` into *the largest context this model + machine
can hold* (the KV is a hard budget, not a free setting), and turns a request beyond it into a
**StreamingLLM** policy: keep `sink` attention sinks, slide a window, assign positions inside
the cache. `vipera kv <arch.toml> --target <hw.toml> --model <gguf>` prints the cap and the
flags; `vipera run --stream` arms it.

**The mechanism is llama.cpp's own** `--context-shift --keep N` (verified in
`llama-cli --help`). It is *disabled by default and ships with `--keep 0`* — the zero-sink
configuration StreamingLLM (`2309.17453`) says breaks the model. vipera's contribution is the
budget, the sink choice and the wiring, **not new C++**.

**Measured** on `models/Qwen3-0.6B-Q4_K_M.gguf`, `-c 64` (llama.cpp raises n_ctx to 256),
`-n 1000`, `--ignore-eos`, `-t 4`, `-lm dio`, from llama-cli's own JSON (`tokens_predicted` /
`stop_type`):

| config | tokens_predicted | stop_type | n_keep |
|---|---|---|---|
| default | **243** | `limit` — context full | 0 |
| `--keep 4 --context-shift` | **1000** | `limit` — n_predict reached | 4 |

Same 256-token KV window ⇒ **4.1× more tokens generated before stopping**. End to end through
`vipera run … --stream -- --ignore-eos` the wall time is **4.19 s → 14.35 s** at ~77 t/s, i.e.
the run really reached its request instead of the window edge.

The budget itself (`vipera kv arch/deepseek-v2-lite.toml --target hw/gtx1650.toml --model models/DeepSeek-V2-Lite.Q4_K_M.gguf`):
**cap 11 992 tokens** at f16 (23 984 at `q8_0`); a 32 768-token request is clamped to the cap
and slides, and the clamp is warned about rather than silent.

**Cost, now measured** — the "length, not fidelity" sentence above was an assertion in the last
round; it is now a number. `llama-perplexity` could not express the sliding regime at all
(it accepts `--keep`/`--context-shift` and ignores both, and clears the KV cache every window),
so the patch is one dispatch in `tools/perplexity/perplexity.cpp`: with `--ppl-stride > 0`,
`--context-shift` and a positive `--keep`, a streaming path keeps the cache alive and shifts it
exactly as `llama-cli` does. Both paths score **the same tokens** (identical window geometry),
so the two PPLs differ only in what the model may attend to. `bench/perplexity.py
--compare-stream` runs the pair and reports the gap against the noise instead of hiding it.

`[measured]` wikitext-2 test, `Qwen3-0.6B-Q4_K_M`, `-c 512 -b 512 --ppl-stride 256
--chunks 195`, `-t 4`, sequential (`bench/ppl_stream_sweep.sh`):

| attention | KV window | PPL | ± (std err) | shifts | Δ vs full window |
|---|---|---|---|---|---|
| full window, contiguous | 768 | **20.61** | 0.30 | 0 | — |
| sink + sliding, `--keep 32` | 768 | **20.74** | 0.31 | 135 | +0.13 (+0.64%) — **inside the noise** |
| sink + sliding, `--keep 4` (StreamingLLM) | 768 | **21.45** | 0.32 | 131 | +0.84 (+4.05%) |
| sink + sliding, `--keep 1` | 768 | **21.59** | 0.32 | 130 | +0.97 (+4.72%) |
| full window, contiguous — *different KV budget* | 320 | **21.87** | 0.32 | 0 | +1.26 (+6.1%) |

The last row is the control the earlier draft of this section said would answer "or just use a
smaller window?". It is a plain contiguous window of 320 tokens (`-c 256`, stride 128, 391
chunks), evaluated over a 50 240-token span against the 50 304-token span above — 0.13% shorter,
so the PPLs are comparable to well inside the noise.

**Sink count is not a detail — it is most of the gap.** Against the full window:

* `--keep 4` and `--keep 1` cost **+4.05% (1.9σ) / +4.72% (2.2σ)** against a standard error of
  the difference of 0.44 (≈50 k scored tokens, 195 chunks). Both clear the harness's 1σ screen;
  the 4-sink arm sits just under the 2σ that a two-sided 95% claim needs and the 1-sink arm just
  over, which is why the harness now prints the sigma instead of only a verdict.
* `--keep 32` costs **+0.64% (0.3σ) — inside the noise**: the sliding window becomes
  indistinguishable from the full contiguous one while never stopping (135 context shifts).

So StreamingLLM's 4-sink convention, chosen for generation stability, is **not** the right
number when the thing you care about is fidelity, and `DEFAULT_SINKS = 4` is now documented in
`vipera/longcontext.py` as a convention to sweep rather than an optimum (the default does not
move: one model, one corpus, and 4 is what the stability result was established with).
`--keep 4` and `--keep 32` differ by 0.70 PPL at 1.6σ — suggestive, not conclusive on its own;
the conclusive part is 32 vs the full window, which is the same token set.

**Sink+sliding versus giving up memory — only the 32-sink arm wins.** The 320-slot contiguous
window sits 1.26 PPL (+6.1%, 2.8σ, so significant) above the full 768-slot window, which is the
price of the smaller budget. Against the sliding arms over the same text: `--keep 32` (20.74)
beats it by 1.13 PPL at **2.5σ — significant**, while `--keep 4` (21.45) beats it by only 0.42
at 0.9σ — **inside the noise**. So the honest claim is narrow: with enough sinks, sink+sliding
buys unbounded length without paying for it in fidelity *and* without needing the memory a full
window needs; with four sinks it buys length and is merely no worse. A 20-chunk pass of the
same pairs could not resolve any of this: it gave +4.2% at a difference error of 1.6.

H2O (`2306.14048`) would evict by accumulated attention instead of by recency, keeping the
tokens that matter; llama.cpp still has no code path for it, so it stays `[proposed]`. §E tests
the cheap substitute for it — a signal the decoder gives away for free — and that substitute
loses to recency, which leaves exposing attention scores as the only route left. Tests:
`tests/test_longcontext.py`, `tests/test_perplexity.py`.

## C. File-layer transform (S9 GGUF tensor writer) — **DONE, `[measured]`**

vipera's stated core (T1 reorder / T4 prune-merge in `docs/OPTIMIZER.md`) was a reader only:
`vipera/gguf.py:309 read_gguf` parsed headers and descriptors, and **nothing in `vipera/`
wrote tensor data**. `vipera/gguf_write.py` is that writer, and it implements both operations
the literature asks for as byte moves over already-quantized slabs:

* **reorder** — `experts[layer]` is a permutation: the co-activation "flash layout" of
  Neuralink (`2410.19274`, digest §17 row 3), so a trace-driven access order is a sequential
  read. `vipera rewrite --random SEED` writes the control arm for measuring what it is worth.
* **prune** — `experts[layer]` is a subset: the write side of `vipera prune`, shrinking the
  tensor's expert dimension so the plan becomes a file other runtimes can load.

The key fact that makes this small: the expert dimension of a `(n_embd, n_ff, n_experts)`
tensor is the slowest-varying one, so expert `e` is **one contiguous run of bytes**. No
quantizer, no dequantize/re-quantize round trip, no dependence on the quant format at all —
which is why one writer unblocks every Grade-C paper instead of one writer per format. The
metadata block is copied *verbatim* (hence the reader now records `kv_end`), so no GGUF value
type has to be re-encoded either.

`[measured]` on a byte-patterned fixture where every expert slab is filled with a distinct
byte (`tests/_fixture.py make_gguf_with_expert_data`):

| run | result |
|---|---|
| prune 4 → 2 experts × 2 layers | `8 -> 4` experts, file `174 816 → 109 280` bytes (**exactly 4 slabs**) |
| reorder (permutation) | every slot holds the *expected expert's* bytes, file size unchanged |
| dense tensors | byte-identical to the source |
| `vipera analyze` on the output | reads `experts 2` — the file is loadable, not just short |

Tests: `tests/test_gguf_write.py` (18), `tests/test_cli.py` (rewrite paths).

**Honest expectation:** on a RAM-resident model this is worth **≤0.2%** (`plantodo.md`). It is a
*capability* purchase: the payoff is in the spill / long-context regime, and it is now measured
by what it can produce rather than by what it lacks.
**Now run on the real model** (`[measured]`, `models/DeepSeek-V2-Lite.Q4_K_M.gguf`, 10 364 416 736
bytes, 377 tensors, 26 layers × 64 experts). Three results, and the second is the one that
matters:

| check | result |
|---|---|
| `vipera rewrite --random 7` on the real file | **18.2 s**, 78 expert tensors rewritten, size unchanged (`delta_bytes 0`) |
| llama.cpp loads the rewritten 10.36 GB file | `llama-perplexity … --chunks 2` → **exit 0**, PPL computed |
| that file is *not* the original model | PPL **4.5448 → 671.7939** (148×) |
| **round trip**: shuffle, then shuffle back with the inverse order | every one of the **377 tensors byte-identical to the original** (`bench/verify_rewrite.py`, exit 0) |

The round trip is the proof of losslessness, and it is checked at whole-file scale rather than
on a fixture: permute the expert slabs, then permute them back, and the result hashes equal to
the source for every tensor, with identical metadata and descriptors (`bench/verify_rewrite.py`,
new, 32 s for 20 GB of reads). The check has teeth — pointed at the *shuffled* file it fails and
names exactly the **78** tensors that were edited (26 × `ffn_{down,gate,up}_exps`), nothing else.

**A measured consequence, not a footnote: a reorder is not semantics-preserving on its own.**
PPL 4.54 → 671.79 is what happens when the expert slabs move but `ffn_gate_inp` keeps routing by
slot index — the model is structurally fine (llama.cpp runs it cleanly) and functionally
wrecked. Neuralink's flash layout is therefore only usable once the router's expert columns are
permuted with it, which `rewrite_experts` does **not** do yet; the `--random` arm shipped here is
purpose-built as the *control* for measuring what a layout is worth, and it measures this.

**Still not done:** no reordered file has been shown to *load* in llama.cpp with its semantics
intact, because that needs the router to move too (next step), and the spill-regime baseline
(a model that does not fit RAM) is still unmeasured — the success test for a *useful* reorder
remains: it must move a spill-regime number, not the resident one.

## D. Activation sparsity (PowerInfer-style) — **probed, verdict: blocked upstream**

Every byte-saving paper trades bytes vipera does not need; **only cutting FLOPs/token raises
the ~23 tok/s ALU ceiling**, which is the binding constraint here. This is the one axis with
real headroom, so it got the honest first step: measure whether the sparsity is there before
building anything.

**1. The shipped model has no zeros to skip** (`[measured]`, from the file itself):
DeepSeek-V2-Lite's FFN is **gated** — `blk.N.ffn_gate_exps` + `ffn_up_exps` + `ffn_down_exps`,
26 layers each, `(2048, 1408, 64)` Q4_K — i.e. SwiGLU, whose intermediate is
`silu(gate) · up`. `silu` is never exactly zero, so the quantity PowerInfer skips (ReLU zeros)
**does not exist in this model as shipped**.

**2. The activation measurement, on DeepSeek-V2-Lite** (`[measured]`):
`bench/sparsity.py` reads `llama-imatrix`'s per-input-channel energy
(`llama-imatrix -m models/DeepSeek-V2-Lite.Q4_K_M.gguf -f wiki.test.raw -c 512 --chunks 8 -t 4`,
4096 tokens; 76.8 MB imatrix). Sampled layers:

| layer | experts exercised | dead channels | energy carried by |
|---|---|---|---|
| `blk.1.ffn_down_exps` | 64/64 | **0.0%** | 23.1% of 1408 channels |
| `blk.5.ffn_down_exps` | 64/64 | **0.0%** | 34.3% |
| `blk.13.ffn_down_exps` | 64/64 | **0.0%** | 26.7% |
| `blk.20.ffn_down_exps` | 64/64 | **0.0%** | 23.3% |
| `blk.25.ffn_down_exps` | 64/64 | **0.1%** | 21.5% |
| `blk.1.ffn_down_shexp` (dense) | — | **0.0%** | 4.5% of 2816 channels (top decile: 76%) |

So: **no dead channels anywhere, and each expert's energy spread over a fifth to a third of its
1408 channels.** Energy is *concentrated* (10% of channels carry 40–76%) but concentration is not
sparsity — a sparse kernel needs values it may **skip**, and there are none. The same picture
holds on a dense shipped SwiGLU model (`Qwen3-0.6B`: 0–0.1% dead channels, top decile 40–98%).

Two caveats, because they bound the claim: an imatrix aggregates over tokens, so this is
*static* channel concentration — it cannot show per-token sparsity, which is the property a
sparse kernel actually exploits. And 4096 tokens is a small corpus for 1664 expert slots; the
imatrix `--chunks 1` route for exposing unexercised experts was not needed here because every
expert had energy. Layout was verified rather than assumed: expert `e` is a *contiguous* block
of the `(n_in, n_experts)` tensor (between-expert CV 1.22 vs 0.08 for the strided reading).

**3. The ceiling if the sparsity existed** (`[modelled]`, `vipera/arch.py` on
`arch/deepseek-v2-lite.toml`): active params 2.45 B, of which the **routed-expert FFN is 55.1%**,
**shared experts 18.4%**, attention 15.2%. A *perfect* sparse predictor on the routed slice is
worth at most ~0.5× decode FLOPs (≈2× at best, before memory traffic), and the shared-expert FFN
is dense by construction and cannot be skipped at all.

**Verdict: building D now would be building the wrong thing.** Three preconditions, none met:
(1) the model must be ReLU-converted and **re-trained** (ProSparse / ReLU²-style) on 15.7 B
params; (2) a predictor must be trained on activation traces vipera cannot collect; (3)
llama.cpp needs neuron-aware sparse FFN operators, where vipera's standing rule is env-driven
patches only. Note the asymmetry with A and B: those controlled flags that already existed, so
the win was real on the first run — D has no such flag to control, and measured against the
shipped model its premise fails at the source.

**Cheap falsification, if someone wants to reopen it:** run a dense ReLU-family model
(OPT / LLaMA-ReLU) through the same command and look for genuinely dead channels. If none is
dead there either, the premise fails at the source and no predictor engineering fixes it.

## E. Retention by *surprise*, instead of by attention — **CREATED, MEASURED, VERDICT: NO**

§B left one lever open: H2O (`2306.14048`) evicts by accumulated **attention** instead of by
recency, which is the right *kind* of signal — but llama.cpp exposes no attention scores to a
control process, so H2O needs a runtime change rather than a planner.

The idea this stage tests is whether that change is **necessary at all**: a decoder already
emits a full logit vector per token, so its own `-log p(token | context)` — "surprise" — is a
relevance signal it produces **for free**, at no kernel change, no attention readback and no
extra memory. If it worked, `surprise` would replace H2O with something vipera can ship today.

**The algorithm, as implemented** (a fourth arm in the same streaming patch, selected by
`LLAMA_PPL_STREAM_POLICY`, with `LLAMA_PPL_STREAM_MARGIN` as the one parameter):

1. accumulate each live token's surprise, `Σ -log p(tokenᵢ | context)`, as the window fills;
2. when the window is full, rank the candidates (all but the sinks, **never** the newest token —
   llama.cpp requires the next batch to continue at `last position + 1`, so dropping it makes
   the next decode fail) by accumulated surprise, lowest first;
3. drop the least surprising half.

`surprise-local` is the hybrid: the same ranking, but the newest `margin` positions are never
candidates, so surprise only ever orders the **older** part of the window and local continuity
is guaranteed by position.

**Design — why the arms are comparable.** `recency` is llama-cli's own behaviour (drop the
middle, close the gap by renumbering positions). The other three drop the *same count* of
tokens, at the *same* budget, from the *same* window geometry, and leave positions alone; so
they differ from each other in exactly one thing — which tokens go. `recency-gap` is the
control for the *other* difference (selection vs position layout): oldest-first, positions kept.

`[measured]` wikitext-2 test, `Qwen3-0.6B-Q4_K_M`, `-c 512 -b 512 --ppl-stride 256 --chunks 195
--keep 4 --context-shift` (window 508), 50 304 tokens spanning **same 49 921 scored tokens and
the same 131 context shifts in every arm** — only the evicted tokens differ:

| retention rule | PPL | ± (std err) | Δ vs `recency` | strength |
|---|---|---|---|---|
| `recency` — drop the middle (llama-cli's own) | **21.4478** | 0.318 | — | — |
| `recency-gap` — drop the oldest, positions kept | **21.9076** | 0.327 | +0.460 | 1.0σ (a tie) |
| `surprise-local`, margin 128 | **24.1365** | 0.340 | +2.689 | **5.8σ** |
| `surprise-local`, margin 64 | **26.0370** | 0.364 | +4.589 | **9.5σ** |
| `surprise` — no margin | **31.9334** | 0.451 | +10.486 | **19.0σ** |

The ordering is what makes the verdict robust: the damage is **monotone in how much of the
window surprise is allowed to order**. Suppress it entirely and the policy *is* `recency-gap`
(21.91); let it order the older 380 of 508 slots and it costs 2.69 PPL; protect only the newest
token and it costs 10.49. Margin 128 vs margin 64 is itself 3.8σ, so the parameter is doing real
work — and turning it *up* walks the policy back toward recency, not away from it.

**The mechanism, which is the reusable part:** a KV entry's worth is **not how hard its own
token was to predict, and surprise measures exactly that.** Dropping the *least* surprising
tokens removes the low-information, high-frequency material — function words, repeated syntax,
the boilerplate that makes a context coherent — and keeps rare, "interesting" tokens whose
predictability is low. The quantity that the window actually needs is *how much the future
attends to* an entry (H2O's quantity), and predictability is not a proxy for it.

**So the shortcut is closed, and the direction that remains open is the restrictive one:**
ranking by a signal the decoder gives away for free **loses to ranking by position** at equal
budget on this model and corpus, in proportion to how much freedom it is given. H2O is **not**
refuted by this — it ranks on attention mass, a different quantity — but it can no longer be
evaded: anyone who wants to beat recency must expose attention scores (a patch, exactly like
the perplexity one this stage already ships), not find a cheaper signal.

**A bug this stage found, and why the artifact caught it.** The first build accumulated surprise
only for the pure `surprise` arm, so under `surprise-local` every candidate scored 0.0, the sort
fell back to position order, and the arm **silently became `recency-gap`** — a policy with a
margin parameter that did nothing. The measurement is what exposed it: two *different* policies
produced `PPL 21.9076 ± 0.32696`, identical to five digits. Two policies cannot tie like that,
and the tie was the defect report. The buggy artifact is kept as
`bench/results/archive/ppl-policy-A-surprise-local-BUG.json`; the fix is one condition.

Reproduce (sequential — `docs/FINDINGS.md` §2; ~4 min per arm):

```bash
bash bench/stream_policy_sweep.sh A        # all five arms, or A-surprise-local for just the pair
```

Thread count is not a confound in this table: the same arm scores `PPL 13.9764 ± 1.77128` at
`--threads` default and at `-t 4`, to four digits, dropping the same 764 tokens (3-chunk probe).

**Shipped as an experiment, not as a stage.** There is no `vipera surprise` command and no
planner change: by this repo's own rule, a lever that has not moved a measured number on this
machine stays `[proposed]`. What is shipped is the reproducible arm — the patch's env-driven
policy, `bench/perplexity.py --policy/--margin`, and the sweep — so the next person can re-run
the negative result instead of re-deriving it.

---

## What "done" means here

A, B and C are implemented; D and E are **decided against, with the reason and the number
recorded** — D at the source of its premise, E by direct measurement at the same budget. The same
discipline still binds: **a number without a label is a defect**, and a lever that has not moved
a measured number on this machine is `[proposed]`.

Where each stage stands at the end of this round:

| stage | code | tests | measured on this machine | still open |
|---|---|---|---|---|
| A. launch policy | `vipera/launch.py` | 13 | 1.51–2.12× decode, median 1.74× | first-divergence check |
| B. long context | `vipera/longcontext.py` | 11 + harness | 4.1× more tokens; **+4.05% PPL at 4 sinks, +0.64% (noise) at 32** | H2O eviction |
| C. GGUF writer | `vipera/gguf_write.py` | 18 + 6 CLI + verifier | real 10.36 GB round trip is **lossless** (377/377 tensors) | router must be permuted with the slabs |
| D. activation sparsity | `bench/sparsity.py` (probe only) | 11 | 0.0–0.1% dead channels; FFN is gated ⇒ no zeros to skip | needs ReLU re-training upstream |
| E. surprise retention | patch arm + `bench/perplexity.py` (experiment only) | harness | **loses to recency**: 24.14 (5.8σ) at margin 128, 31.93 (19σ) with no margin | wants attention scores, which llama.cpp does not expose |
