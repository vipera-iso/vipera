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
