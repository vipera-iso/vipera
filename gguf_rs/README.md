# gguf_rs — the optional Rust island for GGUF index reads

`vipera model import|inspect|optimize` parse a GGUF index. Almost all of that
cost is one loop: a GGUF string array stores `[len][bytes]` per entry, so its
end is only known by walking every length, and a real vocabulary is 150k-400k
entries the analyzer never reads.

This crate does that walk in Rust and nothing else. It takes the `mmap` through
the buffer protocol (no copy, whatever the file size) and returns the end
offset. Every GGUF rule -- metadata types, tensor sizes, the fields the analyzer
reads -- stays in `vipera/gguf_fast.py`.

| file | package `gguf` | Python walk | with the island |
|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M | 4138 ms | 36 ms | 4.9 ms |
| Qwen2.5-7B Q4_K_M | 4015 ms | 36 ms | 2.9 ms |
| gpt-oss-120b Q2_K shard 1 (49 GB) | 8234 ms | 62 ms | 6.1 ms |

All three readers return an identical `analyze_gguf` dict on every installed
model; `tests/unit/test_gguf_fast.py` pins that.

## Install

```bash
pip install "vipera-ai[gguf-rs]"     # prebuilt wheel from the release
```

From a checkout, with a Rust toolchain:

```bash
make gguf-rs                          # maturin develop --release into .venv
```

## It is optional

`vipera` imports `_gguf_rs` and keeps the Python walk when it is missing, so a
plain `pip install vipera-ai` is correct and only slower. `VIPERA_GGUF_FAST_PURE=1`
forces the Python path, which is how the parity test compares the two.

## The one `unsafe`

A length prefix can sit at any byte offset, so the walk reads it with
`read_unaligned`; the pointer is only dereferenced after `pos + len <= buffer`,
which is checked on every entry. Decoding, alignment of the payload, and every
other rule is Python's job.
