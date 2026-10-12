# Captured artifacts

Real evidence, small enough to keep in git.

| file | what it is | how it was produced |
|---|---|---|
| `olmoe-1b-7b-511tok.jsonl` | routing trace, one prompt, 511 decode tokens | `python3 -m bench.capture_trace --model olmoe-q4km.gguf --tokens 512 --ngl 0 --threads 8` |
| `olmoe-multitopic-545tok.jsonl` | routing trace, **5 topics concatenated** into one session (545 tokens) | `python3 -m bench.capture_trace --prompts-file prompts.txt --tokens 110 --out ...` |
| `hwprobe-run1.json`, `hwprobe-run2.json` | measured bandwidth + GEMV throughput | `gcc -O3 -march=native -fopenmp -o hwprobe bench/hwprobe.c && ./hwprobe` |

The model is `allenai/OLMoE-1B-7B-0924-GGUF` Q4_K_M (16 layers, 64 experts, top-8).
Both traces are captured with the llama.cpp fork patch in `bench/llama-moe-trace.patch`.
Analysis of these files is in `docs/ROADMAP.md` §8 and §10.
