# Plan: fast GGUF index reader for `analyze_gguf`

## Problem, with the measured number

`analyze_gguf` on the 6.9 GB shard (`q8src-imbf-q3km`) takes **4107.5 ms** (median-3).
`cProfile` attributes 10.6 s (inflated) to one place:

| call | count |
|---|---|
| `gguf_reader._build_fields` -> `_get_field_parts` | 455,312 |
| `gguf_reader._get` | 761,885 |
| `_get_str` | 303,833 |
| `np.memmap.__getitem__` | 2,285,868 |

That is the tokenizer vocabulary (151,936 tokens + 151,387 merges) decoded one
Python object at a time. `analyze_gguf` never looks at the vocabulary.

Where the time is *not*: `vipera serve` startup does not call `analyze_gguf` at
all (measured: `tools/profile_startup.py --no-load` = 76 ms total; import 39.8,
hardware probe 28.0, config 7.9, model resolve 0.1). The 4.1 s lands on
`vipera model add` / `model info` / `model inspect`, which parse the index on
the user's command.

## Goal and gate

- Target: **< 500 ms** on the same 6.9 GB file.
- Gate: the `analyze_gguf` return value must be **identical** to the current
  implementation (`==` on the whole dict) for every model file on this machine.
  Identity is checked by running both readers on the same file, not by eyeballing.
- No new dependency. Same bytes parsed, fewer Python objects built.

## Approach

Parse the GGUF index directly with `mmap` + `struct`, and skip the cost that is
unnecessary:

1. Header: magic, version, `tensor_count`, `kv_count`.
2. KV block: read each key, then the type. Materialize scalars and small
   numeric arrays (<= 4096 items). Skip the value payload for everything else
   (tokenizer strings, huge chat templates, big arrays) by advancing the cursor
   arithmetically; a string array is skipped by walking lengths only, never
   decoding.
3. Tensor infos: name, shape, type, offset. `n_elements = prod(shape)`,
   `n_bytes = type_size * n_elements // blck_size` using `GGML_QUANT_SIZES`
   taken from the installed `gguf` package (a static dict) so the arithmetic is
   bit-identical; unknown type raises, which falls back.

The analyzer needs exactly two shapes: `reader.fields[key].contents()` and
`reader.tensors` (`name`, `shape`, `tensor_type`, `n_elements`, `n_bytes`). The
fast module provides a minimal reader with those attributes, so
`_analyze_single` keeps its logic untouched -- which is what makes "identical
dict" a property by construction rather than a hope.

## Safety

- Env `VIPERA_GGUF_READER=pkg` forces the old reader (used by the parity test).
- Default is fast-then-package: any parse failure (unknown version, unknown
  quant type, truncated file, malformed string) falls back to `GGUFReader`, so
  a file that works today keeps working.
- The import guard message ("gguf is not installed") is preserved when the
  package is the thing that fails.

## Steps

1. `src/vipera/gguf_fast.py`: `read_gguf(path)`, `_Field`, `_Tensor`, `GGUFReadError`.
2. `gguf_analyzer._analyze_single`: open through `_open_reader()`.
3. `tests/unit/test_gguf_fast.py`: synthetic file parity (fast vs package),
   truncated-file error, real-model full-dict parity (skipped when no model).
4. Measure median-3 before/after on the 6.9 GB file; record in
   `OPTIMIZE-techniques.md` only if it passes and the dicts match.

## Not doing

- Not porting the analyzer to Rust/C. The cost is the number of Python objects
  built, not interpreter throughput; a struct-based reader in Python removes it
  without a new build step or FFI boundary.
- Not touching `vipera serve`, which spends none of these seconds.

## Result

| file | before | after | ratio |
|---|---|---|---|
| qwen2.5-1.5b Q4_K_M | 4138 ms | 36 ms | 115x |
| Qwen2.5-7B Q4_K_M | 4015 ms | 36 ms | 111x |
| hermes4-14b Q4_K_M | 4066 ms | 36 ms | 113x |
| q8src-imbf Q3_K_M (6.9 GB) | 4178 ms | 57 ms | 73x |
| gpt-oss-120b Q2_K shard 1 (49 GB) | 8234 ms | 62 ms | 132x |

Gate met: `analyze_gguf` returns an identical dict on all five files (`==` on the
whole value), and `tests/unit/test_gguf_fast.py` pins that. Median-3, same
machine. Target was < 500 ms.

What is left of the 36 ms is the one loop that cannot be skipped: advancing over
~400k string lengths to find the end of the tokenizer arrays.

One correction found while writing the plan: the 4.1 s was never on the `serve`
startup path. `tools/profile_startup.py --no-load` measures 76 ms total there
(import 39.8, hardware probe 28.0, config 7.9, GGUF index 0.1).

## Follow-up (same day): the last loop went to Rust

The remaining ~25 ms was the walk over the tokenizer arrays' length prefixes. A
small PyO3 island (`gguf_rs/`, built with `make gguf-rs`) does that walk on the
mmap through the buffer protocol and returns the end offset; everything else
stays in Python.

| file | package gguf | Python struct | + island Rust |
|---|---|---|---|
| qwen2.5-1.5b Q4_K_M | 4138 ms | 36 ms | **4.9 ms** |
| Qwen2.5-7B Q4_K_M | 4015 ms | 36 ms | **2.9 ms** |
| hermes4-14b Q4_K_M | 4066 ms | 36 ms | **3.6 ms** |
| q8src-imbf Q3_K_M (6.9 GB) | 4178 ms | 57 ms | **3.6 ms** |
| gpt-oss-120b Q2_K shard 1 (49 GB) | 8234 ms | 62 ms | **6.1 ms** |

All three paths return an identical `analyze_gguf` dict on all five files. The
island is optional: without it the Python walk runs
(`VIPERA_GGUF_FAST_PURE=1` forces that path in tests).
