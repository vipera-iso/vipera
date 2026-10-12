# OPTIMIZER — the model-file optimization layer

> This is **vipera's core positioning**: it does not write a new runtime — it **rewrites the model
> file** so a weak runtime reads less and reads sequentially. The runtime (llama.cpp) is a *consumer*
> of the artifact vipera produces.

---

## 1. Why optimize at the file layer

On the target machine (4 GB VRAM, 15 GiB RAM, DRAM-less NVMe), MoE decode cost is dominated by
**the number of bytes read per token**. Runtime tricks (cache, prefetch) only push toward the I/O
ceiling; they **cannot change** the fact that hot experts are scattered across the disk in the
original file.

Break the problem down:

```
token cost  =  (bytes read from VRAM/RAM)  +  (bytes read from disk)
               └─ fast, unchanged ─┘          └─ SLOW (2 GB/s) — this is what we optimize ─┘
```

**The optimization layer's goal:** reduce *bytes read from disk*, and when a read is unavoidable,
make it **sequential**.

---

## 2. Pipeline

```
 raw safetensors / GGUF
        │
        ▼
 ┌─────────────────┐   (1) analyze
 │  ANALYZER       │   • routing traces (which expert is picked per layer/token)
 │                 │   • expert frequency • stale groups • tensor sizes
 └────────┬────────┘
          ▼
 ┌─────────────────┐   (2) transform (inputs/outputs stated per transform)
 │  TRANSFORMER    │   • expert reorder • prune/merge • block-quant • chunk align
 └────────┬────────┘
          ▼
 ┌─────────────────┐   (3) pack
 │  PACKER         │   • optimized .gguf artifact
 │                 │   • `*.vipera.json` = residency manifest
 └────────┬────────┘
          ▼
   runtime (llama.cpp) reads the manifest → knows what lives where
```

---

## 3. Residency manifest — `*.vipera.json`

The contract between the optimizer and the runtime. Planned contents:

```jsonc
{
  "vipera_version": 1,
  "source": { "model": "deepseek-v2-lite", "quant": "Q4_K_M", "sha256": "..." },
  "target": { "vram_mb": 4096, "ram_mb": 11264, "disk": "nvme-dramless" },
  "placement": {
    "resident_vram": ["attn.*", "shared_experts.*", "norm.*"],
    "resident_ram":  ["hot_experts/*"],        // e.g. the 60% hottest experts
    "spill_disk":    ["cold_experts/*"]        // read on demand
  },
  "expert_order": "by_frequency",              // order within the file
  "routing": { "trace_tokens": 4096, "topk": 6, "hit_potential": 0.71 },
  "kv_cache": { "type_k": "q8_0", "type_v": "q8_0", "pages": 256 }
}
```

The runtime reads the manifest ⇒ it does **not** have to probe or guess at startup, avoiding a blind
choice of `-ngl`/`--n-cpu-moe`.

---

## 4. Transforms in detail

### T1. Expert hot/cold reorder
- **Input:** routing traces (from P2) + the original file.
- **Output:** a file whose experts are ordered by frequency; hot experts **adjacent on disk**.
- **Why it wins:** a DRAM-less NVMe reads sequentially far better than randomly. Many hot experts
  adjacent ⇒ **one sequential read**, not N scattered reads.
- **Risk:** if the expert distribution shifts by prompt domain, an order tuned for one task is worse
  for another ⇒ reorder from *many* traces, not a single one.

### T2. Residency manifest
- **Input:** model + machine config.
- **Output:** `*.vipera.json` (section 3).
- **Why it wins:** it removes the guesswork at run time; placement is explicit ⇒ less trial and error,
  and it becomes the **input** for the runtime's expert cache.

### T3. Block-wise quantization
- **Input:** safetensors (f16/bf16).
- **Output:** 4-bit block GGUF (independent blocks, mmap per block).
- **Why it wins:** fewer bytes on disk and in RAM; independent blocks ⇒ each block can be `mmap`ed,
  which pairs well with streaming. (AirLLM uses exactly this idea to cut I/O.)

### T4. Expert pruning / merging
- **Input:** per-expert usage frequency + quality contribution.
- **Output:** fewer experts (drop those almost never selected, or merge similar pairs).
- **Why it wins:** MoE is already sparse ⇒ cutting experts gives a **direct** memory win.
- **Risk:** **silent quality loss**. Always measure perplexity before/after; over-pruning ⇒ roll back.

### T5. KV placement hints
- **Input:** desired context configuration.
- **Output:** manifest entries (KV type, page count).
- **Why it wins:** KV easily exhausts RAM on this machine; stating it explicitly avoids OOM (the
  machine has **no swap**).

### T6. Chunk alignment
- **Input:** model file.
- **Output:** tensors aligned to page/disk boundaries.
- **Why it wins:** avoids over-reading bytes at page edges — important on a DRAM-less disk.

---

## 5. Interfaces

```python
# vipera/optimizer.py (design, not yet implemented)
class Optimizer:
    def analyze(self, model_path: str, traces: list[str]) -> "Analysis": ...
    def transform(self, model_path: str, plan: "Plan") -> "Artifact": ...
    def write_manifest(self, artifact: "Artifact", target: "TargetHW") -> str: ...
```

Planned CLI:

```bash
# analyze: emit traces + expert statistics
vipera analyze models/deepseek-v2-lite-Q4_K_M.gguf --trace bench/prompts.txt -o analysis.json

# optimize: write the artifact + manifest
vipera optimize models/deepseek-v2-lite-Q4_K_M.gguf --target hw/gtx1650.toml -o out/

# output: out/model.vipera.gguf, out/model.vipera.json
```

`hw/gtx1650.toml` describes the target machine (VRAM/RAM/disk) so placement can be computed.

---

## 6. Measurement (mandatory before trusting any transform)

| Transform | How to verify |
|---|---|
| T1 reorder | bytes read/token before–after; hit %; tok/s |
| T2 manifest | placement error (predicted vs actual peak VRAM/RAM) |
| T3 block-quant | perplexity rise ≤ 0.3 vs the original file |
| T4 prune/merge | perplexity on a **diverse question set**, not one prompt |
| T5 KV hints | peak RAM, no OOM |
| T6 align | cold-load time (cold cache) |

**Principle:** every transform must **prove itself** with before/after numbers. Any transform that
does not win on the target machine is **disabled by default** and recorded as a failure — it is not
kept around to look good.
