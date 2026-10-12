# Captured artifacts

Real evidence, small enough to keep in git.

| file | what it is | how it was produced |
|---|---|---|
| `olmoe-1b-7b-511tok.jsonl` | routing trace, one prompt, 511 decode tokens | `python3 -m bench.capture_trace --model olmoe-q4km.gguf --tokens 512 --ngl 0 --threads 8` |
| `olmoe-multitopic-545tok.jsonl` | routing trace, **5 topics concatenated** into one session (545 tokens) | `python3 -m bench.capture_trace --prompts-file prompts.txt --tokens 110 --out ...` |
| `hwprobe-run1.json`, `hwprobe-run2.json` | measured bandwidth + GEMV throughput | `gcc -O3 -march=native -fopenmp -o hwprobe bench/hwprobe.c && ./hwprobe` |
| `olmoe-1b-7b-prefetch-table.txt` | expert prefetch table for the fork patch: 16 layers, 541 experts | `python3 -m bench.prefetch_table bench/results/olmoe-1b-7b-511tok.jsonl -o bench/results/olmoe-1b-7b-prefetch-table.txt` |

The model is `allenai/OLMoE-1B-7B-0924-GGUF` Q4_K_M (16 layers, 64 experts, top-8).
Both traces are captured with the llama.cpp fork patch in `bench/llama-moe-trace.patch`.
Analysis of these files is in `docs/ROADMAP.md` §8 and §10.

## Fork patches for llama.cpp

vipera drives llama.cpp through two small local patches. Neither changes what the
model computes; they only expose, and then act on, the router's expert choice.

| patch | what it does | enable with |
|---|---|---|
| `bench/llama-moe-trace.patch` | dumps one JSON line per decode token, listing the experts each MoE layer selected | `LLAMA_MOE_TRACE=<path> llama-cli ...` |
| `bench/llama-moe-prefetch.patch` | pulls predicted experts into the MoE expert cache (`--moe-cache-mib`) while the previous layer computes | `LLAMA_MOE_PREFETCH=<table> llama-cli ... --moe-cache-mib N` |

Apply both with `git apply` against the commit they were cut from. The prefetch
table comes from `bench/prefetch_table.py`, which reads a trace and asks the same
question `vipera prune` does: which experts hold a layer's routing mass?

On the captured OLMoE trace a 0.95-mass table needs **541 experts across 16 layers**
(mean 33.8 of 64 per layer). That is the honest scale of the idea: routing here is
concentrated enough to help a cache that already holds roughly half the experts, and
too flat to help a cache that holds far fewer. The patch reports the same thing at
runtime as `prefetch: placed = N, useful = M`, so a run states its own verdict.

