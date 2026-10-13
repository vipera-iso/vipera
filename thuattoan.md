# thuattoan.md — Toàn bộ thuật toán tối ưu để chạy model LLM trên máy yếu

> Tài liệu này gom **hết ngữ cảnh** vào một chỗ: máy đích, model đích, bài toán, và từng thuật toán
> vipera dùng để chạy được một LLM. Nguồn là code thật trong `vipera/` và `bench/`, cùng các tài liệu
> `docs/ALGORITHMS.md`, `docs/PROPOSALS.md`, `docs/OPTIMIZER.md`, `docs/ROADMAP.md`.
>
> **Quy ước độ tin cậy** (giữ đúng như repo):
> **đo được** = số đo trên máy thật, có lệnh tái lập · **mô hình hoá** = số do vipera tính, không phải
> chạy thật · **đề xuất** = chưa hiện thực · **công bố** = số của người khác, chưa kiểm chứng lại.

---

## 0. Ngữ cảnh

### 0.1 Máy đích (đo 2026-10-12)

| Thành phần | Thông số | Vì sao quan trọng |
|---|---|---|
| CPU | Intel i5-11300H, **4 nhân / 8 luồng**, tới 4.4 GHz | MoE compute rớt về CPU rất chậm |
| RAM | **15 GiB**, ~10 GiB dùng được, **không swap** | model phải ≤ ~11 GiB mới nằm trọn trong RAM |
| GPU rời | NVIDIA **GTX 1650 Max-Q, 4 GB VRAM**, compute 7.5 (Turing) | chứa được attention + KV + vài expert, **không chứa cả model** |
| GPU tích hợp | Intel Iris Xe (Tiger Lake) | không có API hợp nhất với NVIDIA ⇒ bỏ |
| Đĩa | WD SN530 NVMe, **DRAM-less** | đọc tuần tự tốt, đọc ngẫu nhiên/burst kém |
| Hệ điều hành | Arch Linux, không sudo, Python 3.14 | cài đặt phải kiểu root-free |

Hằng số trong `hw/gtx1650.toml`:

| Hằng số | Giá trị | Trạng thái |
|---|---|---|
| `ram_bw_gbs` | 44.0 | **đo** (40–46 GB/s, `bench/hwprobe.c`) |
| `disk_bw_gbs` | 8.0 | **đo** (7.75–9.1 GB/s, O_DIRECT) |
| `cpu_gflops` | 110.0 | **đo** (GEMV ALU-bound, 8 luồng) |
| `vram_bw_gbs` | 192.0 | **datasheet, chưa kiểm chứng** |
| `ram_headroom_mb` | 5120 | **đo** (giữ lại phần OS + môi trường thực chiếm) |
| `vram_headroom_mb` | 512 | dành cho KV + activation |

### 0.2 Model đích

| Giai đoạn | Model | Tổng / active | Q4_K_M | Vừa 11 GiB RAM? |
|---|---|---|---|---|
| P0–P2 | **DeepSeek-V2-Lite** | 15.7B / 2.4B | ~9.5–10.4 GB | ✅ |
| stretch | Qwen3-30B-A3B | 30.5B / 3.3B | ~18.6 GB | ❌ (phải stream) |

### 0.3 Bài toán và nguyên lý duy nhất

```
token cost = (byte đọc từ VRAM/RAM)  +  (byte đọc từ disk)
             └── nhanh, không đổi ──┘    └── CHẬM — đây là thứ tối ưu ──┘
```

`tokens/s ≈ băng thông / (số byte trọng số phải đọc mỗi token)`.

Ba trần trên máy này (đo 2026-10-13) — cái nào thấp hơn thì ràng buộc:

| Trần | Tính | Kết quả |
|---|---|---|
| Số học (ALU) | 2.4B active × 2 = 4.8 GFLOP/token ÷ 110 GFLOP/s | **~23 tok/s** ← ràng buộc thật |
| Băng thông RAM | 1.2 GB/token ÷ 44 GB/s | ~37 tok/s |
| Đọc đĩa khi trượt cache | 1.2 GB ÷ 8 GB/s | **~6.7 tok/s** |

⇒ Khoảng cách giữa "nằm trọn RAM" (~20 tok/s) và "trượt cache ra đĩa" (~6.7 tok/s) là **~3×**.
Kết luận thiết kế: **tăng hit rate của cache expert là mục tiêu số một**; "stream cho nhanh" không
thu hẹp được khoảng cách đó.

### 0.4 Hai chế độ của vipera

| Chế độ | Lệnh | Bản chất |
|---|---|---|
| Mô hình hoá | `analyze`, `plan`, `schedule`, `trace`, `prune`, `sweep`, `score` | số học trên kích thước tensor + mô phỏng decode. **Không nạp model.** |
| Chạy thật | `run` | đọc manifest → dịch sang cờ llama.cpp → chạy binary thật → **đo** throughput/RSS/disk |

---

## 1. Bản đồ pipeline

```
 raw GGUF / safetensors
        │
        ▼
  (1) analyze   → Inventory: tách byte thành resident / router / shared-expert / other / expert
        │
        ▼
  (2) plan      → *.vipera.json: chỗ ở của từng nhóm (VRAM / RAM / spill disk)
        │
        ▼
  (3) schedule  → mô hình hoá bytes/token + tok/s, chạy cache policy & alloc
        │
        ├── trace   → entropy routing, ceiling cache tĩnh cùng cỡ
        ├── prune   → keep-set expert theo routing mass (P4 có cổng + rollback)
        └── prefetch → bảng expert dự đoán cho patch llama.cpp
        │
        ▼
  (4) run       → áp manifest vào llama.cpp và CHẠY THẬT, đo kết quả
```

---

## 2. Thuật toán A — Phân loại tensor (`vipera/analyze.py`)

Mọi tensor được chia vào 5 nhóm, quyết định bởi tên tensor:

| Nhóm | Nhận dạng | Có mặt mỗi token? |
|---|---|---|
| resident | `blk.N.*` còn lại (attention, norm) | có |
| router | `ffn_gate_inp.weight` | có |
| shared expert | `*_shexp*` | có |
| other | `token_embd`, `output`, `output_norm` | có |
| routable expert | `*_exps*` (và không phải `_shexp`) | **không** — chỉ khi router chọn |

```
always_resident_bytes = resident + router + shared_expert + other
expert_bytes          = tổng tensor *_exps
n_experts             = mode của shape[2] trên các tensor 3 chiều
n_layers              = metadata block_count, fallback = max blk index + 1
```

Đây là thứ biến "tối ưu model" thành một bài toán **chỉ trên kích thước byte** — không cần đọc dữ liệu
tensor (parse 9.65 GiB header + descriptor mất ~0.24–0.37 s).

---

## 3. Thuật toán B — Quy hoạch chỗ ở, greedy "always-needed first" (`vipera/schedule.py:227`)

```
1. resident_total = tổng always-resident
2. resident_vram = min(resident_total, usable_vram)
   resident_ram  = min(resident_total − resident_vram, usable_ram)
   resident_spill = resident_total − resident_vram − resident_ram     → cảnh báo nếu > 0
3. cache_bytes = (usable_vram − resident_vram) + (usable_ram − resident_ram)
4. slots = cache_bytes // expert_each              ← số expert vừa cache
5. expert_pool  = expert_each × (moe_layers × n_experts)
   expert_spill = max(0, expert_pool − cache_bytes)
```

Độ phức tạp `O(#tensor)`. Thuộc tính:

- **Chỉ dựa trên kích thước** — không xếp hạng expert nào là "hot" cho tới khi có trace thật; manifest
  ghi rõ điều đó ở `placement.expert_order_pending_trace`.
- Trong `vipera/manifest.py`, biến thể này xuất ra `*.vipera.json`. Nó **không** ép được điều runtime
  thực sự làm (ví dụ llama.cpp giữ input embedding trên CPU), nên tài liệu ghi rõ giới hạn này.
- `ram_headroom_mb` phải là số **đo**, không phải số chung: bản cũ để 1536 MiB ⇒ planner gọi một model
  15.9 GiB là "nằm trọn" trong khi máy không có đủ RAM.

---

## 4. Thuật toán C — Expert cache: `lru` / `lfu` / `hot` (`vipera/schedule.py:308`)

Khóa cache là cặp `(layer, expert)`. Ba chính sách:

| policy | chọn victim | ghi chú |
|---|---|---|
| `lru` | key ít được dùng gần đây nhất | baseline; đây là thứ router MoE chạm tuần hoàn nên rất dễ thrash |
| `lfu` | key có tần suất thấp nhất | **mọi miss đều được admit** → burst one-off xô lệch hot set |
| `hot` | LFU + **admission filter** + **pinned hot set** + **aging** | P1 (SAC) — mặc định tốt nhất |

### 4.1 `hot` gồm ba cơ chế

**(a) Admission filter** (`_admit`):
```
nếu cache đầy và freq[key_mới] <= freq[victim]:  → BYPASS (không chiếm slot)
ngược lại:                                        → evict victim, admit key
```
Expert "one-hit wonder" bị chặn, đếm riêng ở `bypasses`. Đây là thực hành CDN chuẩn, không có trong
cache MoE thông thường.

**(b) Pinned hot set** (`_age`): ghim `pin_fraction × capacity` expert tần suất cao nhất
(**mặc định 50%**) để không bị đuổi, kể cả khi đang đầy.

**(c) Aging**: cứ mỗi `aging_interval = 8 × capacity` lượt truy cập thì **chia đôi mọi counter tần số**,
rồi ghim lại top-K. Con số **8** là **đo được**, không phải chọn bừa:

| aging | ratio trên routing tĩnh | ratio sau khi routing shift |
|---|---|---|
| 2× | 0.80 (quá mạnh) | — |
| không aging | — | 0.70 (không theo kịp) |
| **8×** | **> 0.9** | **> 0.9** |

### 4.2 Đo được (mô hình hoá trên Zipf)

```
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --cache-policy lru
python3 -m vipera schedule arch/glm-5.3-flash.toml --target hw/gtx1650.toml --quant q4-experts --cache-policy lfu
```

| policy | hit rate | disk bytes/token | tok/s |
|---|---|---|---|
| LRU | 21.5% | 3.11 GiB | 0.544 |
| **LFU (P1)** | **32.6%** | **2.66 GiB** | **0.620** |

→ **+11.1 điểm hit, +14% throughput, không mất chất lượng.**

Trên **trace thật OLMoE** (511 token, 64 expert, top-8), so với ceiling cùng cỡ:

| cache | ceiling | LRU | LFU | `hot` |
|---|---|---|---|---|
| 10% | 42.7% | 0.0% | **0.97×** | 0.93× |
| 25% | 71.1% | 0.75× | **0.97×** | 0.93× |
| 50% | 94.1% | 0.99× | 0.99× | 0.98× |

**Ba chỉnh sửa quan trọng mà trace thật mang lại:**
1. trên routing **tĩnh**, **LFU thắng `hot`** (aging chỉ đáng giá khi phân bố *dịch chuyển*);
2. **cấp phát theo tầng không giúp gì** (§8 ROADMAP);
3. pruning **không miễn phí** (§8 dưới).

---

## 5. Thuật toán D — Cấp phát slot theo tầng: marginal water-filling (`_reallocate`)

```
marginal value của 1 slot ở layer l = xác suất routing của expert "hot kế tiếp" của l
lặp `capacity` lần:
    cấp slot cho layer có marginal lớn nhất (argmax)
```

Vì mọi marginal **không tăng** theo số slot đã cấp, tham lam argmax này cho **lời giải tối ưu chính
xác** của bài toán phân bổ tách rời (separable knapsack). Giải lại mỗi `alloc_interval = 4 × capacity`.
Khi các tầng giống nhau nó thu về chia đều ⇒ không tốn gì.

**Đo được: per-layer KHÔNG giúp** trên dữ liệu thật ⇒ mặc định là `global` (`--cache-alloc global`).
Sau mỗi lần cấp phát lại, tầng nào vượt quota mới thì bị cắt bớt victim cho tới khi vừa.

---

## 6. Thuật toán E — Trace, entropy, và "ceiling" đối chứng (`vipera/trace.py`)

### 6.1 Định dạng trace
JSONL, mỗi dòng = 1 token: `{"t": <step>, "layers": [[e,...], ...]}` — mỗi entry là một **MoE block**
(khối dense đầu tiên không có router nên không xuất hiện).

`load_trace` / `validate(routing, n_layers=arch.moe_layers, n_experts, top_k)`:
- `moe_layers = n_layers − leading_dense_layers` — chính vì model thật (DeepSeek-V2-Lite có 27 block,
  1 block dense đầu) làm trace có 26 tầng; đây là thứ tự từng chặn mọi lệnh trace.

### 6.2 Độ lệch routing — entropy
```
H = entropy phân bố chọn expert của một tầng
H ≈ 1.0  → routing đều, KHÔNG cache nào giúp được
```

### 6.3 Static cache ceiling — chống tự lừa
`global_cache_ceiling(routing, slots)` = **oracle tĩnh**: hit rate cao nhất mà một cache **đúng bằng
`slots`** có thể đạt trên chính routing đó (chọn tập expert phổ biến nhất, giữ nguyên suốt).

Mọi hit rate của policy được báo kèm tỉ lệ `policy / ceiling` **ở cùng kích thước cache**. Trước đây
ceiling được tính ở cỡ cache khác, khiến policy đúng trông đẹp hơn cận thật của nó.

---

## 7. Thuật toán F — Prefetch theo dự đoán routing

### 7.1 Vì sao dự đoán được
MoE **layer-local**: expert của tầng n+1 phụ thuộc hidden state của tầng n. Router chỉ lộ lựa chọn khi
tầng đó sắp chạy ⇒ llama.cpp gốc **nạp theo nhu cầu**, mọi upload đều lộ (không che được latency).

### 7.2 Bảng dự đoán (`vipera/prefetch.py`)
Bảng = **per-layer mass keep-set** lấy từ trace — **cùng phép đo với pruning**, nên lời khuyên prune và
bảng prefetch không bao giờ mâu thuẫn.

```
# comment
<layer>\t<expert>,<expert>,...     expert khả năng cao nhất đứng trước
```
`build_table(routing, target_mass=0.95, top=N)`; `top` cắt mỗi tầng (cache nhỏ thì bắt buộc cắt).

### 7.3 Patch llama.cpp (`bench/llama-moe-prefetch.patch`) — **không fork core**
Patch chỉ thêm:
- `moe_cache_lru::reserve(il, e)` — chiếm slot **không** upload (lấy từ đầu LRU như một miss),
- đọc bảng từ `LLAMA_MOE_PREFETCH`, điều khiển `LLAMA_MOE_PREFETCH_DEPTH` (mấy tầng trước) và
  `_TOP` (cap mỗi tầng),
- trong lúc tầng hiện tại compute, `tensor_set_async` các expert dự đoán của tầng kế **cùng nhóm
  device**,
- tự chấm điểm: `placed` / `useful` (dự đoán được router chọn trước khi bị evict) / `extra upload`.

vipera **không sửa thêm** llama.cpp: `vipera run --trace` học bảng, ghi ra file, rồi arm qua env. Nếu
binary không có patch thì runner **nói rõ** (`has no MoE prefetch support`) thay vì để một lần chạy
trông như "prefetch không giúp gì".

### 7.4 Đo được (trace 199 token bắt từ chính DeepSeek-V2-Lite)

| cấu hình | gen t/s | verdict của patch |
|---|---|---|
| Vulkan + prefetch (cache 512 MiB) | 1.30 | 474/5440 (**8.7%**), +28.8 GiB upload |
| CUDA + prefetch (cache 1024 MiB) | 2.00 | 935/8320 (**11.2%**), +44.0 GiB upload |

**Kết luận trung thực:** prefetch **làm chậm đi** ở đây. Trace quá ngắn ⇒ bảng sai. Trên OLMoE, bảng
0.95-mass cần **541 expert / 16 tầng** (trung bình 33.8/64) — tức routing đủ tập trung để giúp cache
**đã giữ ~nửa số expert**, và quá phẳng để giúp cache nhỏ hơn nhiều.

---

## 8. Thuật toán G — Pruning expert

### 8.1 Pruning theo khối lượng (`prune.py:91`)
```
với mỗi layer:
  đếm freq[e] trên trace
  sắp expert theo freq giảm dần
  lấy tiền tố nhỏ nhất sao cho Σfreq / Σtất_cả ≥ target_mass   (mặc định 0.95)
```
Tham lam, `O(E log E)` mỗi tầng, chỉ cần histogram — miễn phí.

**Đo được** trên DeepSeek-V2-Lite, trace 199 token bắt trong phiên này:
```
python3 -m vipera prune /tmp/ds_trace.jsonl --arch arch/deepseek-v2-lite.toml --target-mass 0.95
→ mass-based keep-set: 1325 of 1664 experts (79.63% of the pool), retaining 95.3% of routing mass
```
(Trên trace khác, `report.md` ghi 1292/1664 = 77.6% — cùng kết luận, chỉ khác trace.)

Bảng sparsity của trace này:

| sparsity | mass giữ được | nếu chia đều | gain |
|---|---|---|---|
| 5% | 14.9% | 5.0% | +9.9 |
| 10% | 25.0% | 10.0% | +15.0 |
| 25% | 49.5% | 25.0% | +24.5 |
| 50% | 76.5% | 50.0% | +26.5 |
| 75% | 93.0% | 75.0% | +18.0 |

⇒ **Phải giữ ~78–80% số expert mới đủ 95% routing mass.** Lợi ích bộ nhớ nhỏ hơn vẻ ngoài — pruning
**không miễn phí**.

### 8.2 P4 — Gated pruning với rollback (`prune.py:261`)
Cổng xét theo thứ tự, mỗi tầng độc lập; tầng trượt cổng thì **giữ nguyên toàn bộ expert** (hỏng cục bộ,
không âm thầm phá cả model):

```
no_routing       → tầng không có tín hiệu routing
nothing_to_prune → keep-set đã bằng cả tầng
mass_unreachable → cỡ keep-set > max_keep_fraction
drift            → drift đo được của tầng > delta
else             → prune
```

Công thức cổng chuẩn (`docs/PROPOSALS.md` §5):
```
routing_mass(K) = Σ_{e∈K} freq_e / Σ_e freq_e ≥ 1 − τ
drift_proxy(K)  = ‖f(x;K) − f(x;full)‖ / ‖f(x;full)‖ ≤ δ
```
Lưu ý: vipera **không chạy model**, nên `drift` là **đầu vào** từ probe, không tự tính. `delta` mà
không có dữ liệu drift là **lỗi**, vì cổng drift không thể đánh giá.

---

## 9. Thuật toán H — Mô hình chi phí decode (`simulate`, `schedule.py:608`)

Với mỗi token, đi qua từng tầng:

```
read       = resident + (expert hit) × expert_each + (expert miss) × expert_each
disk_read  = resident × f_disk + (expert miss) × expert_each
secs       = tier_seconds(resident)                      # chia theo f_vram / f_ram / f_disk
           + cached / bw(ram) + missed / bw(disk)

tier_seconds(n) = n × ( f_vram/bw(vram) + f_ram/bw(ram) + f_disk/bw(disk) ) / 1e9
```

Cộng thêm **embeddings** stream 1 lần mỗi token. Compute floor:

```
compute_s = 2 × active_params / cpu_gflops        # 2 FLOP mỗi tham số
```

Hai cách ghép I/O với compute:

| chế độ | công thức | ý nghĩa |
|---|---|---|
| có prefetch (`prefetch=True`) | `time = max(io, compute)` | I/O **che** sau compute |
| không prefetch | `time = io + compute` | nối tiếp |

Cuối cùng: `ms_per_token = forward_ms / draft_tokens`, `tok_s = 1000 / ms_per_token`.

`--eval-tail <f>` chỉ đo hit rate trên **f cuối** của trace (cửa sổ sau shift), để policy thích nghi
chậm không được "cứu" bởi các token trước đó.

**Đây là số MÔ HÌNH HOÁ** — scheduler luôn in ra dòng "all values MODELLED".

---

## 10. Các knob tối ưu tầng này

| Knob | Cờ | Tác dụng |
|---|---|---|
| Lượng tử | `--quant fp8\|q4\|q4-experts` | `q4-experts` = 4-bit **chỉ cho expert**, giữ attention/norm chính xác hơn |
| KV cache | `--kv-bits 16\|8\|4` | KV `q8_0` gần như miễn phí về chất lượng |
| Prune | `--prune-experts <p>` | giữ phần `p` số expert mỗi tầng |
| Bỏ tầng | `--keep-layers N` | bỏ tầng từ cuối |
| Speculative/MTP | `--draft-tokens γ` | 1 forward ra γ token |
| Cache policy | `--cache-policy lru\|lfu\|hot` | chọn victim |
| Cấp phát | `--cache-alloc global\|per-layer` | water-filling |
| Prefetch | `--no-prefetch` | tắt để so sánh nối tiếp |

`vipera sweep` chạy một tập cấu hình cố định qua đúng hàm mục tiêu này rồi in cạnh nhau.

---

## 11. Tối ưu ở tầng FILE (T1–T6) — **định vị cốt lõi, chưa hiện thực**

`docs/OPTIMIZER.md`: runtime chỉ đẩy tới trần I/O; nó **không thay đổi được** việc expert hot bị rải
rác trên đĩa trong file gốc.

| Transform | Input | Output | Lợi trên máy yếu |
|---|---|---|---|
| **T1** reorder hot/cold | routing trace | file sắp lại | expert hot **liền nhau** ⇒ 1 lần đọc tuần tự thay vì N lần rải rác |
| **T2** residency manifest | model + máy | `*.vipera.json` | runtime biết chia VRAM/RAM/disk thay vì đoán `-ngl` |
| **T3** block-wise quant | safetensors | GGUF 4-bit | block độc lập ⇒ `mmap` từng block |
| **T4** prune/merge expert | tần suất dùng | ít expert hơn | nhỏ hơn ⇒ vừa RAM |
| **T5** KV placement hints | config | manifest | biết KV nằm đâu, mấy bit (máy **không swap**) |
| **T6** chunk alignment | file model | căn theo page/đĩa | tránh đọc thừa ở mép page (đĩa DRAM-less) |

Nguyên tắc bắt buộc: mỗi transform phải **tự chứng minh** bằng số đo trước/sau. Transform nào không
thắng trên máy đích thì **tắt mặc định** và ghi nhận là thất bại.

---

## 12. Đề xuất P2 / P3 — **chưa hiện thực**

### P2 — Tier-aware bit water-filling (TAW)
```
min  Σᵢ fᵢ · bytesᵢ(bᵢ)
s.t. Σᵢ σᵢ · (Bᵢ − bᵢ) ≤ ε          (ngân sách chất lượng)
     bᵢ ∈ {2,3,4,5,6,8,16}
```
Gán bit theo thứ tự giảm dần của `(byte tiết kiệm) / (σᵢ · số bit bỏ)` — knapsack tham lam `O(n log n)`.
Ý tưởng: expert có hit rate cao thì cần **ít bit hơn** (dù sao cũng rẻ để đọc), trọng số luôn-resident
stream từ đĩa thì cần **nhiều bit hơn**.
*Tự phê bình của chính repo:* `σᵢ` (chất lượng mất đi mỗi bit) không tách rời được, nên P2 là heuristic
khoác áo Lagrangian. Ước lượng +10–25% tok/s ở perplexity bằng nhau.

### P3 — Energy-aware tiering (EAT)
```
min  α·T(C) + β·E(C)     s.t. P(C) ≤ P_max , Σ_{e∈C} S_e ≤ M_fast
E(C) = Σ_disk S_e · joules_per_byte(tier)      # DRAM ≪ SSD
```
SSD offload tốn **5–12× năng lượng/token so với DRAM** (**công bố**, chưa kiểm chứng). Thêm **cổng
năng lượng** để không đánh đổi năng lượng lấy tốc độ khi có trần công suất. Đây có thể là đề xuất
**quan trọng nhất nhưng ít bằng chứng nhất**, vì nó dựa vào số của người khác.

---

## 13. Áp manifest vào runtime — `vipera run` (đã hiện thực)

Mapping nhỏ và kiểm tra được:

| Trường manifest | Cờ llama.cpp |
|---|---|
| mọi expert ở RAM | `--n-cpu-moe <n_layers>` |
| `expert_cache_vram_bytes` | `--moe-cache-mib <MiB>` |
| `resident_vram_bytes` (attention/norm) | `-ngl <n_layers>` |
| `kv_cache.type_k` / `type_v` | `-ctk` / `-ctv` |

Hai thứ runner **không giả vờ**:
1. build **không có GPU backend** ⇒ không tôn trọng được phần VRAM → cảnh báo + chạy `-ngl 0`;
2. llama.cpp **không có** per-expert disk streaming ⇒ plan có expert spill bị báo là **không map được**
   (dùng mmap paging, chi phí mà hit rate mô hình hoá không mô tả).

Ngoài ra runner:
- `--compare`: chạy **default llama.cpp** và bản manifest liền nhau, in delta + speedup;
- `--trace` / `--prefetch-table`: học bảng prefetch và arm patch qua env;
- tự thêm `-v` khi arm prefetch, vì thống kê cache/prefetch chỉ log ở mức INFO;
- phát hiện GPU backend bằng cách quét `libggml-*.so` cạnh binary;
- phát hiện patch prefetch bằng cách quét **cả thư viện chia sẻ** (patch nằm trong `libllama.so`, không
  phải stub `llama-cli`).

---

## 14. Số đo thật (cùng cấu hình: ctx 512, 32 token, 8 luồng, model 9.65 GiB, expert ở CPU)

| build | gen t/s | prompt t/s | peak RSS | disk read | wall |
|---|---|---|---|---|---|
| **CPU-only** | **12.80** | 9.6 | 12369 MiB | 2886.5 MiB | 7.9 s |
| CUDA | 6.90 | 10.2 | 10337 MiB | 404.0 MiB | 8.2 s |
| Vulkan | 3.40 | 5.0 | 10110 MiB | 4196.7 MiB | 16.9 s |

Nhận xét trung thực:
- **CPU-only thắng** — vì model không vừa 4 GB VRAM, expert nằm ở CPU, GPU chỉ tăng tốc attention và trả
  giá PCIe mỗi expert. CUDA nhanh gấp ~2× Vulkan nhưng vẫn dưới CPU.
- Cache `2789 MiB` theo manifest **không vừa VRAM thật** (Vulkan lỗi allocate, CUDA lỗi tạo context) ⇒
  phải giảm xuống 1024 qua pass-through `--`.
- `disk read` nhiễu mạnh theo page cache; `prompt t/s` nhiễu vì prompt ngắn.
- `--compare` trên CPU build: default 12.20 → manifest (`-ctk/-ctv q8_0`) 14.90 t/s = **1.22x**.

---

## 15. Interface

### 15.1 CLI
```
vipera analyze  <model.gguf> [--json]
vipera plan     <model.gguf> --target hw/gtx1650.toml [--out DIR]
vipera arch     <arch.toml>
vipera schedule <arch.toml> --target <hw.toml> [--quant ...] [--cache-policy lru|lfu|hot]
                [--cache-alloc global|per-layer] [--trace T] [--eval-tail F] [--no-prefetch]
vipera trace    <trace.jsonl> --arch <arch.toml> [--target <hw.toml>] [--json]
vipera prune    <trace.jsonl> --arch <arch.toml> --target-mass 0.95 [--json]
vipera prune-gated <trace.jsonl> --arch <arch.toml> [--max-keep-fraction F] [--delta D] [--drift J]
vipera sweep    <arch.toml> --target <hw.toml>
vipera score    <arch.toml> --target <hw.toml>
vipera run      <model.gguf> --target <hw.toml> | --manifest M.vipera.json
                [--compare] [--trace T] [--prefetch-table P] [--dry-run] [--json R]
                [--binary BIN] [-- <cờ llama.cpp>]
```

### 15.2 Manifest `*.vipera.json`
```jsonc
{
  "vipera_version": 1,
  "source":  { "architecture": "deepseek2", "n_layers": 27, "n_experts": 64 },
  "target":  { "name": "gtx1650-laptop", "usable_vram_mb": 3584, "usable_ram_mb": 13824 },
  "placement": {
    "resident_vram_bytes": 833058816,
    "expert_cache_vram_bytes": 2925037568,
    "expert_ram_bytes": 6602323968,
    "expert_spill_disk_bytes": 0,
    "expert_cache_fraction": 0.307,
    "expert_order": "uniform",
    "expert_order_pending_trace": true      // chưa có trace ⇒ chưa xếp hạng được expert hot
  },
  "kv_cache": { "type_k": "q8_0", "type_v": "q8_0" }
}
```

### 15.3 Trace JSONL
```
{"t": 0, "layers": [[20,5,42,22,46,45], [7,59,42,61,60,35], ...]}
```
Mỗi entry = một **MoE block** (khối dense đầu không có router).

### 15.4 Bảng prefetch
```
# llama.cpp MoE prefetch table
<layer>\t<expert>,<expert>,...
```

---

## 16. Trạng thái & hạn chế

| Thuật toán | Trạng thái | Bằng chứng |
|---|---|---|
| A. Phân loại tensor | ✅ | `analyze` |
| B. Quy hoạch chỗ ở (greedy) | ✅ | `plan`, `schedule` |
| C. Expert cache LRU/LFU/`hot` | ✅ | +11 điểm hit, +14% tok/s |
| D. Water-filling theo tầng | ✅ | **không giúp** → mặc định `global` |
| E. Trace + entropy + ceiling | ✅ | `global_cache_ceiling` |
| F. Prefetch theo trace | ✅ (qua patch, env) | 8.7–11.2% hữu ích |
| G. Pruning mass + P4 gated | ✅ | 79.6% pool giữ 95.3% mass |
| H. Mô hình chi phí decode | ✅ (mô hình hoá) | `vipera sweep` |
| T1–T6 (tầng file) | ❌ định vị, chưa code | `docs/OPTIMIZER.md` |
| P2 tier-aware bits | ❌ đề xuất | ước lượng |
| P3 energy-aware tiering | ❌ đề xuất | số công bố |
| CPU–GPU co-exec, KV offload | ❌ chưa | — |

**Hạn chế phải luôn nói kèm:**
1. Mọi số của `schedule` là **mô hình hoá**; scheduler không mô hình hoá compute GPU và hiệu suất kernel.
2. Routing mặc định là **Zipf mô hình hoá**, chỉ thay bằng trace thật khi truyền `--trace`.
3. Năng lượng và chất lượng **không** được mô hình hoá ⇒ P2/P3 phụ thuộc đầu vào vipera chưa tạo được.
4. `vram_bw_gbs = 192` là **datasheet**, chưa đo (một toolkit CUDA root-free và build `libggml-cuda` đã
   có, nhưng chưa chạy probe băng thông VRAM).
5. Các số đo GPU là **một lần chạy mỗi cấu hình**, chưa lấy trung bình; `disk read` nhiễu theo page cache.
6. `sweep` và `prune` **không** chứng minh chất lượng — chúng chỉ đếm routing mass; muốn nói về chất
   lượng phải đo perplexity/benchmark (chưa làm).
