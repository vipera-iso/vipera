# External review of vipera

**Pinned commit:** `08f087b` — *Narrow the claim to what was measured, and stop calling a model a measurement* (2026-10-13).
**Working tree:** `master`, clean apart from this document; 9 commits after `3079bb5` (*Put the repository entirely in English*).
**Review language:** the original review was written in Vietnamese on 2026-10-13 from the README and the docs tree; §2–§6 are the verification pass against the code at the pinned commit.

Numbers below are labelled **reported** (quoted from `README.md` / `bench/results/`) or **re-measured here** (executed during this verification). Nothing in this document is a new performance claim.

---

## 1. Verdict

**Research prototype, not a production tool.** The idea — optimise at the *file* layer rather than in the runtime — is sound and clearly differentiated, and the project's measurement discipline is unusually honest. But the core promise, rewriting tensor data, is not implemented: what exists today is a planner/manifest, a simulator, and a runner that drives `llama.cpp`.

---

## 2. What the original review claimed, and what holds against the code

### 2.1 `placement.expert_order_pending_trace` — confirmed, and more precise than the review

`vipera plan` has no `--trace` option (`vipera/cli.py:546-550`: only `model`, `--target`, `--out`), and `cmd_plan` calls `plan(inv, target)` without `sharpness` (`vipera/cli.py:97-101`).

The accurate statement is not "the planner is not wired to traces" but:

> **`expert_order` is a dead label.** It has no consumer, and the manifest has no schema to express an order in the first place.

- Written at `vipera/manifest.py:166-167`, unconditionally (`expert_order_pending_trace: True`).
- Read by: an assertion (`tests/test_manifest.py:70-71`) and prose (`README.md:255`, `report.md:201`, `algorithms-digest.md:140,568`). **No code path consumes it.**
- `placement` holds only per-tier byte **totals** (`vipera/manifest.py:158-168`) — resident, expert cache, expert RAM, spill. There is no per-expert list, so an expert *order* cannot change a single number.
- `run.plan_flags` (`vipera/run.py:139-178`) reads `expert_cache_vram_bytes`, `resident_vram_bytes`, `expert_spill_disk_bytes` and the KV types — never `expert_order`.

Consequently, adding `--trace` to `plan` would flip one string (`"uniform"` → `"by_frequency"`) and one boolean, change no measurement, and be read by nothing. See §5.

Routing traces *are* consumed where they can do work: `schedule --trace` (`vipera/cli.py:574`), `run --trace` (`vipera/cli.py:646`), `prune` / `prune-gated` (`vipera/cli.py:603-623`).

### 2.2 Tensor data is never rewritten — confirmed

- `vipera/gguf.py:1-4`: *"Minimal GGUF v3 reader. Reads only the header, metadata and tensor descriptors — never the tensor data."*
- The only writer in `vipera/` and `bench/` is `vipera/prefetch.py:81` `write_table`, which writes a prefetch table (text), not weights.
- No `safetensors`, no `struct.pack`, no `"wb"` anywhere in `vipera/` or `bench/`.
- `vipera/prune.py:314,318` emits JSON keep-lists, not a rewritten model file.

So there is no expert reorder, no block-wise requantisation, no pruned GGUF. The transform the repo delivers (T2, per `vipera/manifest.py:3`) is **placement metadata**, not tensor bytes.

### 2.3 `vipera run` is a flag wrapper — confirmed, but the review undersold it

`vipera/run.py:1-23` states the mapping and that *"llama.cpp is treated as an external binary and is never patched or rebuilt"*:

| manifest field | llama.cpp flag |
|---|---|
| MoE layers resident in RAM | `--n-cpu-moe` |
| expert cache in VRAM | `--moe-cache-mib` |
| attention / resident weights | `-ngl` |
| `kv_cache.type_k` / `type_v` | `-ctk` / `-ctv` |

Plans with experts spilled to disk are reported as **unmappable** rather than approximated (`vipera/run.py:21-22,168-173`).

What the review missed: this half *measures*, it does not only drive.

- `_Sampler` polls `/proc/<pid>/status` and `/proc/<pid>/io` for peak RSS and cumulative disk read (`vipera/run.py:256-303`).
- Throughput is parsed from llama-cli's own line (`vipera/run.py:49-51,225-235`).
- `compare()` (`vipera/run.py:421-449`) runs the plain default against the manifest-driven command and diffs them.
- `format_compare` **refuses to print a speedup when either run failed** (`vipera/run.py:488-494,524-531`) and flags an OOM kill as an invalid comparison.

That is measurement discipline, not a thin wrapper.

### 2.4 Trace and prefetch depend on patching llama.cpp — confirmed, with a limit

- `bench/llama-moe-trace.patch` and `bench/llama-moe-prefetch.patch` exist.
- `bench/capture_trace.py` requires the `LLAMA_MOE_TRACE` fork and reports a clear error if no file was produced, rather than writing an empty trace.
- `vipera/run.py:107-130` `supports_prefetch()` greps the binary and its libraries for `LLAMA_MOE_PREFETCH` and warns *"will be ignored"* instead of letting the run look like a prefetch that did not help.

**Limit of the claim:** the patches are optional extras. The basic `run` path needs neither, and `vipera/run.py:11` states the core is never patched or rebuilt. "Depends on llama.cpp" is true for the trace + prefetch path, not for the tool as a whole.

---

## 3. What the original review missed

1. **Dead code.** `sharpness` (`vipera/manifest.py:89`) is passed by no caller, so the `"by_frequency"` branch at `vipera/manifest.py:166` is unreachable from the CLI — `plan` always emits `"uniform"`. `expert_order_pending_trace` is in turn read only by prose and one test (§2.1): it is a status marker for a human, not a signal the tool acts on.
2. **`run` is under-rated.** Baseline-vs-manifest comparison plus real RSS/disk/throughput measurement (§2.3) makes it the half of the project that produces evidence, not a mere controller.

Two smaller corrections to the review's facts: the suite is **227 tests, all passing** at this commit (re-measured: `python -m pytest -q` → `227 passed`, 3.9 s; CI runs `unittest discover` on Python 3.11–3.14 and is stdlib-only by design).

---

## 4. Corrections to the original review

| Original wording | Verified wording |
|---|---|
| Planner is still simple and size-based: it cannot rank hot experts without a routing trace (acknowledged in `placement.expert_order_pending_trace`) | True, and weaker than it sounds: even *with* a trace, `plan` would not change — `expert_order` is a dead label (no consumer, no schema for an order). Traces do reach `schedule`, `run` and `prune` |
| `vipera run` is just a wrapper around llama.cpp flags | It maps flags *and* measures throughput, peak RSS and disk read, with a baseline comparison that refuses to report a speedup on a failed run |
| Depends heavily on llama.cpp, fragile under upstream changes | True for trace capture and prefetch only; both are optional and detected at run time |
| 225 tests | 227 tests at `08f087b`, all passing |

---

## 5. Follow-ups (each a separate commit, not part of this review)

1. **Remove or repurpose the dead `sharpness` parameter** (`vipera/manifest.py:89,166`) — a few lines, low risk, keeps the manifest honest by construction.
2. **Manifest schema v2: per-expert tier assignment plus a consumer.** This is what would make "rank hot experts from a trace" real. It is a design change, not a wiring change: today `placement` carries only totals, and llama.cpp exposes no per-expert placement flag (`vipera/run.py:21-22`), so a v2 manifest needs a consumer to be more than a label. The design and the project already say this: the T2 contract specifies per-expert lists (`docs/OPTIMIZER.md:58-67`, `resident_vram: ["attn.*", "hot_experts/*"]`) that v1 never implements, `docs/FINDINGS.md:205` records that "the manifest represents more than the runtime can execute", and `README.md:30` notes there is no per-expert placement for llama.cpp to execute. The work is the *consumer*, not the discovery. The per-expert path that *does* have a consumer already exists — `run --trace` → `prune.plan_from_trace` → prefetch table (`vipera/prefetch.py:37-50`) → patched llama.cpp.

Writing "now wired" into this document after only adding a `--trace` flag would recreate exactly the documentation-versus-code gap this review criticises. Neither follow-up has been done as of `08f087b`.

---

## 6. Conclusion

Unchanged from the original review: **vipera is worth following** — clear idea, exceptional documentation, and a refusal to report a number it did not measure (it corrects its own numbers in `report.md:296` and in the commit history). It is not production software, and the file-layer transforms that would make it distinct are not implemented yet.

For someone interested in running LLMs on weak hardware, the repo is worth reading for *how it measures and plans*, not for a speed-up it cannot yet deliver.

### Reproduction

```bash
grep -rn expert_order --include='*.py' --include='*.md' .   # writer, one assert, prose
grep -rn 'def write\|struct.pack\|safetensors' vipera bench/ # one text-table writer
python -m pytest -q                                          # 227 passed
```
