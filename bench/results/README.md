# Captured artifacts

Real evidence, small enough to keep in git.

| file | what it is | how it was produced |
|---|---|---|
| `olmoe-1b-7b-511tok.jsonl` | routing trace, one prompt, 511 decode tokens | `python3 -m bench.capture_trace --model olmoe-q4km.gguf --tokens 512 --ngl 0 --threads 8` |
| `olmoe-multitopic-545tok.jsonl` | routing trace, **5 topics concatenated** into one session (545 tokens) | `python3 -m bench.capture_trace --prompts-file prompts.txt --tokens 110 --out ...` |
| `hwprobe-run1.json`, `hwprobe-run2.json` | measured bandwidth + GEMV throughput | `gcc -O3 -march=native -fopenmp -o hwprobe bench/hwprobe.c && ./hwprobe` |
| `olmoe-1b-7b-prefetch-table.txt` | expert prefetch table for the fork patch: 16 layers, 541 experts | `python3 -m bench.prefetch_table bench/results/olmoe-1b-7b-511tok.jsonl -o bench/results/olmoe-1b-7b-prefetch-table.txt` |
| `ppl-stream_k{1,4,32}.json`, `ppl-stock-c256.json` | sink+sliding vs a full window vs a smaller contiguous window (`docs/REWRITE.md` §B) | `bash bench/ppl_stream_sweep.sh` |
| `ppl-policy-A-*.json` | the eviction-policy experiment: `recency`, `recency-gap`, `surprise`, `surprise-local` (§E) | `bash bench/stream_policy_sweep.sh A` |
| `archive/ppl-policy-A-surprise-local-BUG.{json,log}` | the build where `surprise-local` scored every candidate 0.0 and so silently *was* `recency-gap` — two "different" policies landing on the same PPL to five digits is what exposed it (§E) | kept as the defect report, not as a result |
| `algo-measure.json` | the four registered algorithms on real artifacts: the Q4_K dequantiser's round-trip check over 12288 real weights, GPTQ in both norms, and the OLMoE trace's modular objective | `python3 -m bench.algo_measure` |
| `moe-spill.json` | the spill regime, three arms: the default OOM-killed, `-nr` running, and the gate arm with a bounded anonymous RSS. Includes `disk_mib_per_token`, which is what cancelled the expert-cache project | `python3 bench/moe_spill.py --arm all` |

The model is `allenai/OLMoE-1B-7B-0924-GGUF` Q4_K_M (16 layers, 64 experts, top-8).
Both traces are captured with the llama.cpp fork patch in `bench/llama-moe-trace.patch`.
Analysis of these files is in `docs/ROADMAP.md` §8 and §10.

## Fork patches for llama.cpp

vipera drives llama.cpp through four small local patches. None of them changes what
the model computes: two expose, and act on, the router's expert choice, one makes
`llama-perplexity` able to express the streaming regime at all, and one changes which
memory an expert tensor lives in — same values, different layout.

| patch | what it does | enable with |
|---|---|---|
| `bench/llama-moe-trace.patch` | dumps one JSON line per decode token, listing the experts each MoE layer selected | `LLAMA_MOE_TRACE=<path> llama-cli ...` |
| `bench/llama-moe-prefetch.patch` | pulls predicted experts into the MoE expert cache (`--moe-cache-mib`) while the previous layer computes | `LLAMA_MOE_PREFETCH=<table> llama-cli ... --moe-cache-mib N` |
| `bench/llama-ppl-stream.patch` | `--ppl-stride` + `--context-shift` + `--keep` score a sliding window through `llama-perplexity` (stock ignores them and clears the cache every window) | `bash bench/ppl_stream_sweep.sh`; eviction policy via `LLAMA_PPL_STREAM_POLICY` (`recency`, `recency-gap`, `surprise`, `surprise-local`) and `LLAMA_PPL_STREAM_MARGIN`. Cut against llama.cpp `e8a3943`; re-cut it with `git -C ~/dev/llama.cpp diff -- tools/perplexity/perplexity.cpp` rather than editing it, or `git apply` will refuse |
| `bench/llama-cpu-repack-moe.patch` | skips repacking the **3D MoE expert tensors only**, so a model whose expert pool exceeds RAM streams from the file instead of being OOM-killed; every 2D weight is still repacked. Commit `cd4a060` in `~/dev/llama.cpp` | `GGML_CPU_REPACK_MOE=0 llama-cli ...` |

Apply them with `git apply` against the commit they were cut from. `moe-spill.py`'s
`control` and `no-repack` arms need **no** patch at all — `-nr` is stock — and the
gate arm measured the same speed within noise, so the patch is a cleaner memory
footprint, not a faster run. The prefetch
table comes from `bench/prefetch_table.py`, which reads a trace and asks the same
question `vipera prune` does: which experts hold a layer's routing mass?

On the captured OLMoE trace a 0.95-mass table needs **541 experts across 16 layers**
(mean 33.8 of 64 per layer). That is the honest scale of the idea: routing here is
concentrated enough to help a cache that already holds roughly half the experts, and
too flat to help a cache that holds far fewer. The patch reports the same thing at
runtime as `prefetch: placed = N, useful = M`, so a run states its own verdict.

