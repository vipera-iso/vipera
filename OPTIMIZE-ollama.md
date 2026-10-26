# OPTIMIZE: Vipera vs Ollama — ledger

Quy ước: **số đo thắng văn bản**, gate Δ > 5% (xem `docs/benchmarks.md`).
Mọi lever phải **đo trước / đo sau**, giữ hay revert đều ghi lại.

## Baseline (Ollama 0.40.8, 2 vòng A B A B, median, mỗi engine một mình máy)

Model: `LFM2.5-8B-A1B-Q4_K_M`, ctx 65536, temp 0, seed 42, n_batch 512.

| metric | Ollama | Vipera | Δ (V−O) |
|---|---|---|---|
| decode t/s | **50.4** | 44.7 | −12.6% |
| e2e ngắn (non-stream) | **44.8** | 41.9 | −7.0% |
| e2e dài (non-stream) | **31.73 s** | 33.51 s | −5.6% |
| prefill qua TTFT | 223 t/s | 218 t/s | +2.2% → **hòa** (dưới gate) |
| TTFT prompt dài | 30.58 s | 31.25 s | **hòa** |

## Nguyên nhân gốc (đã truy tới code)

3 lỗi liên kết khiến planner **dự trù KV 16×** và offload thấp:

1. **`gguf_analyzer._RECURRENT_TENSOR_MARKERS`** `("ssm_","linear_attn","kda","mamba","conv1d")`
   không khớp `shortconv.*` của LFM2.5 → `recurrent_layers=0` → `hybrid=False`
   → `kv_layers = total = 24` (thật: **6**, `[2,6,10,14,18,21]` — khớp "6 layers" của llama.cpp).
2. **`gguf_analyzer._field_first_int`** trả `value[0]` = **0** cho mảng per-layer
   `lfm2moe.attention.head_count_kv` = `[0,0,0,8,…]` (layer conv = 0, layer attention = 8)
   → planner fallback `or n_heads` = **32** (thật: **8**).
3. **Ống dẫn**: `serve_cmd.py:153` `plan(m["meta"], …)` với `m = registry.get_model(...)` —
   registry meta chỉ có `{size_bytes, n_layers, n_params, context_length, quant, architecture}`
   → không có hình học → planner rơi vào heuristic cuối
   `(65536/8192)×1.5×(0.5/2.0)` = **3.000 GB** (khớp banner).

Chuỗi số: `layer_total = 0.20009 + 3.000/24 = 0.32509` → `safe_vram 3.34 / 0.32509 = 10.24` →
**gpu_layers 10** (khớp banner `10/24`).

Sau sửa (kv_layers 6, n_kv_heads 8, head_dim 64): KV = 0.1875 GB → `per_layer 0.03125` →
`layer_total 0.23134` → `3.34 / 0.23134 = 14.4` → **gpu_layers 14**.

## Lever đã đo — `tools/measure_knobs.py` (ctx 65536, q4_0, n_batch 512, REPS 3, median)

| label | ngl | decode t/s | VRAM | ghi chú |
|---|---|---|---|---|
| ngl10-plan | 10 | 54.702 | 2528 MiB | = plan hiện tại |
| ngl14 | 14 | **63.290** | 3414 MiB | **+15.7%** |
| ngl18 | 18 | — | — | failed |
| ngl22 | 22 | — | — | failed |
| ngl24 | 24 | — | — | failed |

→ ceiling = **14**, đúng con số planner sẽ ra khi KV được tính đúng.

## Sửa đổi (giữ hay revert sẽ ghi ở cuối)

Hai sửa trong `src/vipera/gguf_analyzer.py`, không đụng `planner.py`:

1. `_RECURRENT_TENSOR_MARKERS`: `"conv1d"` → `"conv"` (bao gồm `shortconv.*` của LFM2.5).
   Layer conv không có tensor K/V nào, nên trước đó chúng bị bỏ qua hoàn toàn:
   `recurrent_layers=0` → `hybrid=False` → `kv_layers = n_layers`.
2. `_field_first_int`: đọc phần tử **khác 0 đầu tiên** thay vì `value[0]`.
   `lfm2moe.attention.head_count_kv` = `[0,0,0,8,…]` (0 ở layer conv), lấy phần tử
   đầu ra **0** → planner `or n_heads` = 32 thay vì 8.

**Chuỗi số quyết định offload** (`planner.plan`, dòng `per_layer_kv_gb`):

```
per_layer_kv_gb = kv_cache_gb / kv_layers      ← kv_layers triệt tiêu ở đây
             = 2 × n_kv_heads × head_dim × ctx × bytes_per_elem / 1GiB
```

nên **fix 2 (n_kv_heads) là thứ đổi số layer**, fix 1 là thứ làm con số KV + RAM đúng:

| | trước (meta registry cũ) | sau |
|---|---|---|
| đường tính | heuristic cuối `(65536/8192)×1.5×(0.5/2.0)` | hình học thật |
| kv_cache_gb @65536 q4_0 | 3.0000 GB | **0.1875 GB** |
| per_layer_kv_gb | 0.1250 | **0.03125** |
| layer_total_gb | 0.20009 + 0.125 = 0.32509 | 0.20009 + 0.03125 = **0.23134** |
| `int((3.6418−0.3)/layer_total)` | 10.24 → **10** | 14.44 → **14** |

### State: entry registry cũ không có hình học

Không cần sửa ống dẫn — `model_cmd._read_meta` **đã** copy `kv_layers / n_kv_heads /
head_dim / hybrid_attention / layer_bytes` vào registry; entry `LFM2.5` thêm ngày
2026-10-17 (có 6 key) và `qwen2.5-0.5b` là bản **cũ**, nên planner rơi vào heuristic.

Đã làm mới qua đúng `_read_meta` (backup: `~/.vipera/registry.json.bak-before-geometry`):

| model | kv_layers | kv_heads | head_dim | layer_bytes |
|---|---|---|---|---|
| LFM2.5-8B-A1B-Q4_K_M | 6/24 | 8 | 64 | 205 511 584 |
| qwen2.5-0.5b-instruct-q4_k_m | 24/24 | 2 | 64 | 10 300 544 |
| NousResearch_Hermes-4-14B-Q4_K_M-2 | 40/40 | 8 | 128 | 198 001 664 |

`qwen2.5-0.5b` trước đó也被 heuristic gắn **3.00 GB KV cho model 0.5B** (vô lý);
nay 0.1875 GB → `gpu_layers 24` (Full GPU).

## Banner trước/sau (chạy thật, `vipera serve start --ctx 65536`)

```
trước: layer offload 10/24 · kv cache (q4_0) [ OK ] · 3.00 GB   (không có dòng hybrid)
sau:   layer offload 14/24 · kv cache (q4_0) [ OK ] · 0.19 GB
       hybrid attention 6/24 kv layers · flash attention ON · VRAM 3406/4096 MiB
```

VRAM 3406 MiB khớp đúng số `ngl14` đo được ở bảng lever (3414 MiB), tức planner
đạt đúng trần mà card chịu được; không cần ladder fallback (tải thẳng 14).

## Test

- **Mới (GUARD)**: `test_conv_only_layers_mark_hybrid_attention`,
  `test_head_count_kv_array_skips_zero_layers` (mở rộng `_write_gguf_multi` với
  `extra_kv` để phát mảng GGUF), `test_kv_estimate_charges_only_kv_layers_on_hybrid`
  (chốt 0.1875 GB) — cả 3 xanh.
- `test_gguf_analyzer.py`, `test_planner.py`, `test_registry.py`, `test_model_cmd.py`: xanh.
- **6 fail có sẵn, không phải do sửa này**: `test_serve_cmd` ×4, `test_chat_cmd`,
  `test_bench_cmd` — do thay đổi **chưa commit** thêm `spec=spec_decoder(...)`
  (`bench_cmd`/`chat_cmd`/`serve_cmd`) và `engine.py:168` so `OptionInfo <= int`,
  trong khi stub test chưa nhận kwargs `spec`. Kiểm chứng: `git show HEAD` không có
  `spec=` ở ba file đó, và `tests/test_serve_cmd.py` không bị sửa.

## Quan sát (ngoài phạm vi lần này)

- `model_cmd._read_meta` ghi chú "The planner budgets VRAM per layer from this"
  về `layer_bytes` — **đã lỗi thời**: `planner.py` cố ý dùng lại
  `size_bytes / n_layers` (comment trong đó giải thích `layer_bytes` thiếu chính
  xác vì buffer compute CUDA của llama.cpp tăng theo số layer offload).
  `layer_bytes` vẫn được lưu nhưng không th affecting budget.
- `GGUFReader` mất ~6.2 s vì phải đi qua toàn bộ KV (tokenizer ~128k chuỗi) mới
  tới tensor-info. Không gọi `analyze_gguf` ở hot path.
- **Upgrade path chưa giải quyết**: mọi entry registry tạo trước khi `_read_meta`
  có hình học vẫn rơi vào heuristic. Cách làm hôm nay là migrate state; nếu cần
  tự chữa thì nên có `vipera model refresh` (hoặc lazy backfill trong `get_model`).

## Thử nghiệm — A/B sau sửa (cùng protocol, 3 vòng, median)

Nguồn: `/tmp/opencode/result_{OLLAMA,VIPERA,VIPERA_AFTER}.json`.

| metric | Ollama | Vipera trước | Vipera sau | sau vs trước | sau vs Ollama | gate 5% |
|---|---|---|---|---|---|---|
| e2e ngắn (wall) | 1.4197 s | 1.5323 s | **1.3535 s** | **+13.2%** | **+4.9%** | sát ngưỡng, **chưa đạt** |
| e2e ngắn (tok/s) | 45.08 | 41.77 | **47.28** | +13.2% | +4.9% | như trên |
| e2e dài (wall) | 31.746 s | 33.541 s | **31.754 s** | **+5.6%** | −0.02% | **hòa** |
| decode (tok/s) | **50.38** | 44.73 | 48.74 | **+9.0%** | −3.25% | **hòa** (dưới gate) |
| TTFT prompt dài | 30.588 s | 31.239 s | **29.935 s** | +4.4% | +2.2% | hòa |
| prefill dài (t/s) | 214.8 | 203.3 | **214.8** | +5.7% | ±0.0% | hòa |

**Quyết định: GIỮ.** Đây là thắng thật, đo được, và nó xóa đúng khoảng cách mà
baseline chỉ ra: e2e dài từ −5.6% lên hòa tuyệt đối, prefill từ −4.4% lên bằng,
decode từ −12.6% lên −3.25% (dưới gate), e2e ngắn từ −7.0% lên +4.9%.

**Nhưng chưa đạt mục tiêu "nhanh hơn Ollama >5%":**

- Ngắn: +4.9% — thiếu 0.1 điểm phần trăm. Trong sai số 3 vòng (best-of-3 cho
  +6.4%), nhưng theo median thì **không qua gate**; ghi là hòa sát ngưỡng.
- Dài: hòa (31.754 vs 31.746 s). Chênh lệch nằm ở 64 token decode cuối
  (35 t/s vs 55 t/s vì 3/6 KV layer nằm CPU), chỉ ~0.65 s trên 31.7 s → kể cả
  sửa hết cũng chỉ ~−2%, không thể qua 5%.
- Decode: Ollama vẫn +3.25% (dưới gate → hòa).

**Vì sao không đi xa hơn được bằng offload:** trần cứng là 14 layer (ngl15 OOM,
ngl≥18 fail), bản llama.cpp này **không partial-offload**. Ollama báo
"25/25 layers" nhưng buffer CUDA0 chỉ 1653 MiB (~66 MiB/layer) — nó chia layer
theo ngân sách, thứ Vipera chưa làm được. Hai hướng còn lại:

1. Nâng `llama-cpp-python`/`llama.cpp` lên bản có partial offload theo ngân sách
   (đường duy nhất để bắt chước chiến lược 25 layer của Ollama).
2. `spec_ngram` (WIP chưa commit của user) — sẽ thắng decode quyết định, nhưng
   hiện làm 6 test fail nên chưa thể tính vào kết luận.

**Sửa kèm:** `qwen2.5-0.5b` từ heuristic 3.00 GB KV (vô lý với model 0.5B) →
0.1875 GB → Full GPU 24/24.

## Vòng 2 — llama-server + `-ncmoe` (MoE partial offload)

### Phát hiện

- Binding Python 0.3.36 **không có** `-ncmoe`/`--spec-type`; binary
  `~/.local/lib/ollama/llama-server` (llama.cpp 0.7.0-dev) thì có.
- `-ncmoe N` giữ expert của **N layer đầu** trên CPU. Đo VRAM để xác định chiều:
  `N=24` → 972 MiB (toàn bộ expert CPU), `N=20` → 1844 MiB (+872 MiB = đúng
  layers 20-23), `N=12` → 3412 MiB (+2440 MiB = đúng layers 12-23). Help text
  đúng; guide HF nói ngược.
- `-ub` mới là lever prefill, không phải `-b`:
  ub512 long 30.5 s / decode 59.3 t/s; ub768 26.8 / 55.9; ub896 25.3 / 56.7;
  ub1024 24.7 / 51.3. `-b 1024 -ub 512` = 30.39 s, xác nhận `-b` không giúp.
- `-fit` mặc định **on** với `--fit-target 1024 MiB`; nếu không truyền
  `-fit off`, llama-server tự chỉnh `-ngl`/`-ncmoe` và vô hiệu hóa plan.
- `-ncmoe 24` (all expert CPU) chỉ 40.8 t/s → chiến lược "đẩy hết expert xuống
  CPU" không đủ; phải giữ expert trên GPU.

### Cấu hình thắng

```
llama-server -ngl 24 -ncmoe 12 -b 896 -ub 896 \
  -ctk q4_0 -ctv q4_0 -fa on -fit off
```

VRAM: **3496 MiB** (harness) / **3500 MiB** (end-to-end qua `create_engine`),
còn ~230 MiB headroom trên card 3730 MiB.

### A/B (median 3, cùng protocol)

| metric | Ollama | Vipera server | Δ |
|---|---|---|---|
| e2e ngắn (wall) | 1.4197 s | **1.32 s** | **+7.6%** |
| e2e dài (wall) | 31.746 s | **25.26 s** | **+25.7%** |
| decode t/s | 50.38 | **56.67** | **+12.5%** |

Cả 3 metric vượt gate 5%.

### Code đã sửa

- `gguf_analyzer`: tách `expert_bytes` (chỉ routed `*_exps`),
  `shared_expert_bytes` (`*_shexp`), `expert_layer_bytes` (list theo layer).
  Model thật: 4446.8 MiB routed, 22 layer có expert, layer 0-1 = 0.
- `planner`: thêm `moe_cpu_layers`, `n_ubatch`, `n_layers`, property
  `requires_server`; budget
  `free - 0.15 - non_expert - kv - (0.20 + 0.00022*ubatch)`, chọn suffix layer
  lớn nhất vừa VRAM → `ncmoe=12`; giữ `gpu_layers=14` classic cho fallback.
- `server_backend`: `-ngl n_layers` + `-ncmoe`, `-b max(n_batch, ub)`,
  `-ub`, `-fit off` khi binary hỗ trợ.
- `engine`: `resolve_backend` tự nâng `inprocess` → `server` nếu binary có
  `-ncmoe`; nếu không, xóa cờ MoE và cảnh báo, dùng plan classic. Ladder tăng
  `ncmoe` từng layer rồi mới xuống CPU-only.
- `config`: `n_ubatch` (0 = auto; MoE dùng 896).
- `serve_cmd`: banner hiển thị `layer offload 24/24` + `moe experts 12/24`,
  backend thật sau khi resolve.
- Registry LFM2.5 đã refresh (backup `/tmp/opencode/registry.backup.json`).

### Test

- Mới: analyzer tách routed/shared; planner fit 12 layer + fallback khi thiếu
  per-layer metadata/VRAM âm; server flags/probe; engine auto-upgrade/ladder.
- Targeted: **132 passed**. Full suite: 6 fail có sẵn từ WIP `spec_ngram`
  (`test_serve_cmd` ×4, `test_chat_cmd`, `test_bench_cmd`), không đổi.
- `ruff check` và `mypy src/vipera`: sạch.

### Việc còn lại

- `vipera tune` (sweep có cache) — biến planner từ heuristic thành đo thật.
- `vipera model refresh` — backfill metadata cho registry cũ.
- Nguồn model thứ hai (ModelScope) chỉ sau khi P0 ổn định; hiện chưa thêm
  dependency.

## Vòng 3 — mixed-quant experts (2026-10-26)

Baseline đo lại trên cùng máy, `--profile auto`, ctx 65536, 3 request, prompt 249 token,
64 token sinh: **65,76 t/s** decode (65,75 / 65,76 / 66,27), VRAM 3494 MiB. Số 56,67 t/s ở
bảng trên đo bằng protocol khác (prompt dài, `tools/measure_knobs.py`), nên chỉ so trong
cùng một bảng.

**Lever:** nén riêng routed experts, giữ attention/KV/routing ở Q4_K_M.

```bash
llama-imatrix -m LFM2.5-8B-A1B-Q4_K_M.gguf -f tools/eval/wikitext_sample.txt \
  -o /tmp/lfm25.imatrix.gguf -c 512 --chunks 6 -t 4
llama-quantize --allow-requantize --imatrix /tmp/lfm25.imatrix.gguf \
  --tensor-type 'ffn_.*_exps=Q3_K' LFM2.5-8B-A1B-Q4_K_M.gguf out.gguf Q4_K_M
```

| | file | expert bytes | expert layer trên GPU | VRAM | decode @64K |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 4,91 GB | 4,45 GB | 12/24 (`-ncmoe 12`) | 3494 MiB | 65,76 t/s |
| experts Q3_K | 3,82 GB | 3,33 GB | **17/24** (`-ncmoe 7`) | 3504 MiB | **78,05 t/s** |

**+18,7%**, hai dải số không chồng nhau. Planner không cần sửa: nó đọc `expert_bytes` nhỏ
hơn và tiêu 1,2 GiB vừa giải phóng vào 5 expert layer nữa.

**Chất lượng = hòa, không phải thắng:** PPL 29,56 ± 1,88 → 28,69 ± 1,81 trên 12 chunk của
một đoạn `wikitext_sample.txt` mà imatrix **không** thấy (imatrix dùng 6 chunk đầu). Không
có imatrix thì không được phép kết luận tương tự — L1a đã cho thấy Q3 requant từ Q4 tệ đi.

**Code đi kèm:** `vipera model optimize --tensor-type PATTERN=TYPE` (repeatable) + từ chối
khi build `llama-quantize` không có cờ đó. `vipera bench` không còn ghi profile
`gpu_layers: 0` vào `tuning.json` (config `profile: lowmem` sinh ra đúng số 0 đó; nó đã ghi
"0 layer, 37,29 t/s" cho máy này trước khi sửa).

**Đã loại bằng số, không viết code:** `-ot/--override-tensor` cho dense. Hermes-4-14B Q3_K_M
có attention 1192 MiB/40 layer, FFN 4856 MiB. Giữ toàn bộ attention + KV trên GPU: @8K
q8_0 = 1856 MiB → còn 1874 MiB cho FFN = 15 layer (hiện tại 18); @32K q4_0 = 2599 MiB → 9
layer; @64K không còn chỗ. Thua ở mọi ctx card này chứa được.

**Còn lại:** `vipera tune` (sweep ngl/ncmoe/ub/ctx có cache) và `vipera model refresh`
(backfill hình học cho registry cũ) — chưa làm.
