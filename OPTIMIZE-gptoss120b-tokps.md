# gpt-oss-120b trên máy này — thuật toán tối ưu và code layer

> Mục tiêu: **decode tok/s nhanh nhất có thể cho `gpt-oss-120b-q2k` trên chính máy
> này**, không phải trên máy ai khác. Mọi con số dưới đây hoặc (a) **measured** —
> đo trên máy này, có lệnh copy-paste lại được, hoặc (b) **published** — số của
> người khác, ghi rõ nguồn và **chưa** kiểm chứng tại đây, hoặc (c) **proposed** —
> chưa làm. Không có mục nào là suy đoán không dán nhãn.
>
> Quy tắc cũ giữ nguyên: **số đo thắng văn bản**, và một đòn bẩy chỉ được ghi vào
> README/benchmarks khi có JSON thô + lệnh lặp lại được + Δ > 5% so với control.

---

## 0. Sự thật nền — đo lại 2026-10-28

| # | Sự thật | Bằng chứng |
|---|---|---|
| F1 | CPU 4 core / 8 thread (AVX-512), RAM **15.41 GiB**, `MemAvailable` ~7 GB, **không swap** | `free -g`, `nproc` |
| F2 | GPU **GTX 1650 4.00 GB**, 3.64 GB trống; card chỉ chứa được **1.39 GiB** phần non-expert của model này | `nvidia-smi`, mục 1 |
| F3 | Model nằm ở **2 shard**: `-00001-of-00002` 49.46 GB + `-00002-of-00002` 13.11 GB = **62.57 GB**; header khai `split.count = 2` | `ls -l`, `GGUFReader.fields` |
| F4 | Phần **routed expert = 61.07 GB (95–97%)**; **non-expert = 1.39 GiB** (attention 0.39 GB + embedding/output 0.94 GB + norms) | `analyze_gguf` hợp nhất 2 shard |
| F5 | 36 MoE layer × 128 expert, top-4 active (1 expert = **13.25 MB**); tên file là `Q2_K` nhưng **tensor expert là MXFP4** (47.94 GB) — `Q2_K` chỉ đúng cho phần nhỏ | `quant_sizes` của analyzer |
| F6 | Analyzer **tự cảnh báo** 29 tensor `ffn_down_exps.weight` "MXFP4, below Q4_K; sensitive to quantization" | `warnings` của analyzer |
| F7 | Vipera cảnh báo khi plan: *"keeps ~57.2 GB of weights on the host and 7.5 GB is available, so ~49.6 GB will be paged in from disk on demand (mmap)"* | log `vipera bench run` |
| F8 | Baseline thật hôm nay: `--ctx 8192`, 112-token prompt → TTFT **117.9 s** (prefill 0.95 t/s), decode **1.59 t/s** (2/36 layer), total 157.5 s | `/tmp/vipera-baseline.log` |
| F9 | Bản đo tốt nhất đang lưu trong `tuning.json`: **2.734 t/s**, `gpu_layers 2`, ctx 8192 | `~/.vipera/tuning.json` |
| F10 | `llama-server` CUDA **đã có sẵn** (`~/llama-bin/llama-b11827`, wrapper `~/.vipera/bin/llama-server`), có `-ncmoe/--n-cpu-moe`, `--moe-cache-mib`, `-ot`, `--spec-type`, `-nr` | `llama-server --help` |
| F11 | `nvcc` **có** ở `~/cuda/bin/nvcc` (CUDA 12.9) — blocker "máy không có nvcc" trong `OPTIMIZE-tokps.md` **không còn đúng** | `nvcc --version` |
| F12 | `~/dev/vipera/bench/llama-moe-prefetch.patch` đã hiện thực **bảng prefetch theo routing** cho MoE cache của llama.cpp, nhưng bản CUDA server đang dùng là binary stock → patch chưa được dùng | `strings` binary, `src/llama-moe-cache.cpp` |

---

## 1. Ngân sách byte mỗi token — vì sao 2.7 t/s và cái gì thực sự chặn

```
mỗi token đọc:  expert active   = 36 lớp × 4 expert × 13.25 MB  ≈ 1.91 GB
                non-expert      = attention + norms + output    ≈ 1.39 GiB
                ─────────────────────────────────────────────────────────
                tổng                                            ≈ 3.3 GB/token
trần RAM  44 GB/s  →  ~13 t/s nếu mọi thứ nằm trong RAM
trần đĩa   8 GB/s  →  ~2.4 t/s nếu mọi thứ phải đọc từ NVMe
đo được            →  1.6–2.7 t/s   ⇒  máy đang ở vùng **đọc từ đĩa / thrashing**
```

Ba hệ quả, và chúng định hình cả bảng xếp hạng bên dưới:

1. **Expert là 95% khối lượng, nhưng chỉ ~58% byte/token** — vì chỉ 4/128 expert
   được đọc. Non-expert tuy bé (1.39 GiB) lại **được đọc mỗi token**, nên nó là
   ~42% byte/token. Đây là lý do đòn bẩy "attention lên GPU" có ý nghĩa dù phần
   này nhỏ (xem §3.1).
2. **Toàn bộ pool expert (61 GB) không thể nằm trong 15 GiB RAM.** 7 GB trống ⇒
   mỗi token phần lớn expert đến từ đĩa. Bất kỳ thuật toán nào **tăng hit rate**
   hoặc **giảm byte/token** đều thắng; thuật toán chỉ "tăng tốc tính toán" thì
   không, vì máy không tính toán nhiều — nó **chờ đĩa**.
3. Vì vậy thứ tự đòn bẩy ở §3 được xếp theo **byte/token và hit rate**, không
   theo "tối ưu kernel".

---

## 2. Root cause đã tìm ra và đã sửa: planner mô tả model bằng **1 shard**

Đây là phát hiện quan trọng nhất của phiên này, và nó giải thích luôn F4 vs F9.

`analyze_gguf()` mở **đúng một file**. Với model 2 shard, nó trả về số của shard 1
(49.46 GB, expert 48.07 GB) trong khi `size_bytes` mà registry lưu lại là **tổng cả
hai shard** (62.57 GB). Planner lấy hai con số đó trừ nhau:

```
non_expert = 62.57 GB − 48.07 GB = 14.5 GB   ← sai
non_expert thật                  = 1.39 GiB   ← đo được
```

`_moe_cpu_layers()` đòi hỏi phần non-expert phải nằm vừa VRAM trước khi nó chịu đặt
expert lên GPU. 14.5 GB > 4 GB ⇒ nó trả `None` ⇒ planner rơi về **classic split
1–2 layer** (F9) hoặc **CPU ONLY**. Toàn bộ phần attention của 36 lớp — 0.39 GB, đủ
chỗ trên card — bị đọc từ RAM mỗi token.

**Đã sửa (code layer):**

| Layer | Thay đổi |
|---|---|
| `src/vipera/gguf_analyzer.py` | Hợp nhất **mọi shard** (`split.count` + mẫu tên `-000NN-of-000MM`, có `resolve()` để đi qua symlink mà `vipera model add` tạo). Cộng `expert_bytes`, `file_size_bytes`, từng phần tử của `expert_layer_bytes`/`ffn_layer_bytes`, `layer_distribution`, `category_sizes`; thêm `split_count` và `split_declared` để một shard thiếu vẫn nhìn thấy được |
| `src/vipera/commands/model_cmd.py` | `size_bytes` lấy **tổng của model** từ analyzer, không phải `path.stat().st_size` (chỉ là shard 1) |
| `src/vipera/planner.py` | Thêm `_moe_experts_on_cpu_plan()` — kế hoạch cho MoE **không đặt được pool expert**: toàn bộ expert ở CPU, toàn bộ attention ở GPU (`-ngl N -ncmoe N`), kèm footprint host đúng. `lowmem` giờ gọi chung hàm này (hành vi `lowmem` không đổi) |

Kết quả trên chính model này (`vipera model add` lại để refresh meta):

```
trước:  note = CPU ONLY (hoặc Hybrid: 2/36 layers)
        -ngl 2  -t 4 -b 512 -ub 512 -fa on -ctk q4_0 -ctv q4_0
sau:    note = MoE: all 36 expert layers on CPU, attention on GPU
        -ngl 37 -ncmoe 36 -t 4 -b 512 -ub 512 -fa on -ctk q4_0 -ctv q4_0
        ram_estimate 58.9 GB | host_resident 2.0 GB (chỉ KV + compute buffer phải thường trú)
```

`ram_estimate` 58.9 GB vẫn đúng và vẫn cảnh báo "sẽ page từ đĩa" — đây không phải
phép màu biến model to hơn RAM thành vừa RAM. Nó chỉ đặt đúng chỗ cái phần **được
đọc mỗi token**.

**Test đã thêm** (`tests/unit/`, 620 test xanh):
`test_split_model_is_analyzed_as_one_model`, `test_split_model_merges_layer_and_category_tables`,
`test_split_model_keeps_what_the_header_declares_missing`,
`test_single_file_model_is_not_reported_as_a_split`,
`test_read_meta_takes_the_whole_model_size_for_a_split`,
`test_shard_discovery_survives_a_symlink_into_the_model_store`,
`test_spill_moe_keeps_every_expert_on_the_cpu_and_attention_on_the_gpu`,
`test_spill_moe_without_room_for_the_dense_side_stays_classic`,
`test_spill_moe_needs_a_card_and_experts`.

---

## 3. Bảng xếp hạng đòn bẩy — thuật toán → code layer

Xếp theo **kỳ vọng thực tế trên máy này** (byte/token là trục chính, §1).

### 3.1 Đặt attention lên GPU cho model spill — **đã làm, đã đo**

* **Thuật toán.** Với MoE, mỗi lớp gồm hai phần: "always-active" (attention, norm,
  router, embedding) và "routed" (expert). Phần always-active **đọc mỗi token**,
  phần routed chỉ 4/128. Vì vậy VRAM nên chứa **always-active của mọi lớp**, không
  phải **vài lớp nguyên vẹn**. Đây đúng là nguyên tắc trong hướng dẫn MoE offload
  của cộng đồng ([Doctor-Shotgun/HF](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)):
  *"assign all of the always-active parameters to the GPU"*, `-ngl 999 -ot exps=CPU`
  ≡ `-ngl 999 -cmoe`.
* **Bằng chứng ngoài.** Như trên + ik_llama.cpp dùng cùng cấu trúc. Vipera đã đo
  trên model khác: LFM2.5-8B-A1B **37.1 → 49.9 t/s** (+35%) khi giữ expert ở CPU
  nhưng đưa attention lên card (`planner.py`).
* **Kỳ vọng ở đây:** bỏ ~1.39 GiB/token khỏi đường host (≈42% byte/token), và trả
  lại 1.39 GiB VRAM đó cho... chính nó (attention nằm trên card, đọc ở 192 GB/s).
  **Đo được: +20% decode (median, nhiễu), +58% prefill, −1 GB VRAM** — xem §4.1:
  phần lớn 42% lý thuyết bị luật chơi “VRAM đang giữ expert slab của 2 lớp” lấy lại.
* **Code layer:** `planner.py::_moe_experts_on_cpu_plan` (đã thêm),
  `backends/server.py` (đã có `-ncmoe`, `-ctc/-ctv`), footprint ở
  `_host_footprint` (đã nối).
* **Gate:** đo bằng `tools/sweep_spill_moe.py` — xem §4. Δ > 5% mới ghi vào docs.

### 3.2 Quant lại phần expert (MXFP4 → Q2_K/IQ2) — **ĐÃ THỬ, KẾT LUẬN ÂM (2026-10-28)**

Đòn bẩy này đã được thực thi và đo. **Không dùng được**: nó nhanh hơn thật nhưng
phá model. Chi tiết cơ chế + số ở §4.3. Tóm tắt tại sao kỳ vọng ban đầu sai:

* **Không thể dùng IQ2/Q2_K cho tensor expert của model này — sai hình học, không
  phải sai cờ.** Expert row rộng **2880**; mọi type họ K/IQ 2-bit dùng block **256**;
  `2880 % 256 = 64 ≠ 0` ⇒ `tensor_type_fallback()` (`llama-quant.cpp:403`) âm thầm
  hạ cấp. Đã chạy thật để xem:
  ```
  llama-quantize --dry-run --tensor-type "ffn_.*_exps\.weight=iq2_xxs" ... q8_0
  ```
  ⇒ **108 cảnh báo** `ncols 2880 not divisible by 256 (required for type iq2_xxs)
  -> falling back to iq4_nl`, và "quant 2 bit" cho ra file **to hơn file gốc**:
  59 659 MiB (4.28 BPW) → **63 856 MiB (4.58 BPW)**. Đây là cái bẫy tệ nhất trong
  cả nhánh này: lệnh chạy exit 0, không lỗi, chỉ nặng thêm 4 GB.
* **Type sub-4-bit duy nhất vừa hình học là `q2_0`** (block **64**, `2880 % 64 = 0`).
  Đã kiểm chứng: override `ffn_.*_exps\.weight=q2_0` áp **108/108** tensor, **0**
  fallback.
* **`--imatrix` là no-op với `q2_0`.** `quantize_q2_0()` (ggml-quants.c:2114) gọi
  `quantize_row_q2_0_ref()` ở **cả hai** nhánh và không hề đọc `quant_weights`;
  hàm ref còn không có tham số weight. Vì vậy "quantize 2-bit **có imatrix**" =
  "quantize 2-bit RTN", đúng thứ mà tài liệu published cảnh báo là sụp chất lượng
  dốc và không đều. Build imatrix cho q2_0 sẽ tốn ~12 h streaming 62 GB mà **không
  đổi một byte nào** — đã bỏ, và đây là lý do.
* **Code layer (nếu sau này hình học đổi):** `src/vipera/optimizer.py`
  (`QuantTarget`, `_quant_bpw`) + flow `llama-quantize` qua `~/.vipera/bin/`.
  Thêm một bước **bắt buộc** trước khi quant: assert `ncols % blck_size == 0` cho
  từng tensor đích, nếu không thì báo lỗi thay vì để llama.cpp hạ cấp im lặng.
* **Ưu tiên:** **bỏ khỏi roadmap** cho model này. Đòn bẩy còn lại là §3.3 (prefetch)
  và §3.5 (prune expert) — cả hai giữ nguyên độ chính xác của trọng số.

### 3.3 "Coach" page cache bằng prefetch theo routing (madvise) — **đòn bẩy rẻ nhất**

* **Thuật toán.** Khi model > RAM và layer ở CPU, llama.cpp mmap nhưng **OS không
  biết expert nào hot**, nên nó evict ngẫu nhiên → page-fault storm. Patch
  **Expert Sniper**: hook eval callback, đọc top-k expert mà router vừa chọn từ
  `t->src[2]`, gọi `madvise(MADV_WILLNEED)` lên đúng dải nhớ của expert đó.
  *"Zero allocation. Zero memcpy. Zero mutex. One syscall per expert slice."*
* **Bằng chứng (published, rất gần regime của máy này).**
  [madvise 15 dòng: **0 → 0.57 t/s**](https://huggingface.co/waltgrace/llama-cpp-expert-sniper)
  trên MacBook 8 GB với model 10.6 GB (stock thrash = 0 t/s); **no-op callback**
  cũng đã 0.46 t/s; **LRU cache 460 dòng chỉ 0.24 t/s** vì nó *cướp* page cache —
  kết luận: *"don't fight the OS page cache — coach it"*. Cảnh báo của chính tác
  giả: trên máy nhiều RAM thì stock nhanh hơn, **chỉ có lợi khi RAM < model và layer
  ở CPU** — đúng máy này (F1/F3).
* **Đã có một nửa ở nhà:** `~/dev/vipera/bench/llama-moe-prefetch.patch` hiện thực
  **bảng prefetch học offline từ routing trace** (mạnh hơn madvise mù: biết expert
  của *lớp sau*), cộng `vipera/prefetch.py` sinh bảng đó. Blocker cũ là binary
  server stock không chứa patch (F12) — mà giờ `nvcc` đã có (F11), nên build được.
* **Code layer:** (a) llama.cpp: `src/llama-moe-cache.cpp` (patch đã viết, cần
  build bản CUDA); (b) gateway: `backends/server.py` truyền env/`--moe-cache` và
  đường dẫn bảng prefetch; `planner.py` bật khi `size_bytes > ram_free`.
* **Gate:** Δ > 10% decode **và** `majflt/token` giảm (số thứ hai là bằng chứng cơ
  chế, không phải trang trí).

### 3.4 Speculative decoding không cần model draft (`--spec-type ngram-*`)

* **Thuật toán.** Sinh nhiều token ứng viên từ chính prefix (n-gram lookup) rồi
  verify song song; khi câu trả lời lặp lại prompt (viết code/diff — đúng ca agent),
  mỗi lần verify được nhiều token với **cùng một lượt đọc expert**.
* **Bằng chứng (measured, trên máy này, phiên trước):** `--spec-type ngram-map-k`
  **6.23 → 8.54 t/s (+37%)** khi prompt được viết lại, **−1%** trên văn xuôi mới;
  config đã có `spec_type`, mặc định `none` vì tác dụng phụ thuộc ca.
* **Kỳ vọng ở đây:** decode 2.7 t/s là regime mà spec decode **có lợi nhất** (mỗi
  token tiết kiệm được ~0.37 s chờ đĩa). Đây là arm `classic+ngram` trong harness.
* **Code layer:** không cần code mới — `config.spec_type` → `server_command()` →
  `--spec-type`. Việc cần làm là **đo trên gpt-oss** và ghi kết quả vào
  `docs/benchmarks.md`.

### 3.5 Prune expert theo routing mass — giảm byte **vĩnh viễn**

* **Thuật toán.** Expert ít được router chọn thì bỏ khỏi file (hoặc hạ precision
  riêng nó). Byte/token giảm vĩnh viễn, hit rate tăng.
* **Bằng chứng (đã có ở nhà):** `~/dev/vipera/bench/llama-moe-prune.patch`,
  `prune_ppl.py`, `bench/results/prune-ppl-indomain.json`; commit mới nhất của repo
  đó — *"Mask the router to measure what pruning costs, and falsify the mass proxy"*
  — tức là **proxy "expert nào nhiều mass thì quan trọng" đã bị chính họ bác bỏ**.
  Bài học: prune phải kèm **permute router**, nếu không thì PPL nổ (đã đo: 4.54 →
  671.79 khi reorder mà không đụng router — `docs/REWRITE.md` §C).
* **Code layer:** `~/dev/vipera/vipera/rewrite.py` + `bench/llama-moe-prune.patch`;
  phía gateway là `model add` sau khi rewrite.
* **Trạng thái: proposed** — chỉ nên làm **sau** 3.2, vì quant đã cắt nửa byte với
  rủi ro chất lượng dễ đo hơn.

### 3.6 Đổi runtime: ik_llama.cpp (`-fmoe`, kernel AVX-512, IQK)

* **Thuật toán.** ik_llama.cpp gộp phần MoE thành **một kernel hợp nhất** và có
  kernel lượng tử riêng; nó cũng hỗ trợ `-fmoe` cho MoE CPU/GPU hybrid.
* **Bằng chứng (published):** báo cáo **~1.9× nhanh hơn cho MoE inference** so với
  mainline nhờ fused MoE + AVX-512 ([issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480)).
  Hướng dẫn MoE offload cũng liệt ik_llama là lựa chọn cho máy "CPUmaxx".
* **Tại máy này:** `~/dev/ik_llama.cpp` đã có source (commit 2026-10-20), **có
  `LLM_ARCH_OPENAI_MOE "gpt-oss"`**, nhưng **chưa build**. CPU là i5-11300H có
  AVX-512 — đúng chỗ ik_llama mạnh.
* **Code layer:** một backend mới trong `backends/` (theo `backends/protocol.py` +
  `backends/registry.py`), vì `llama-cpp-python` không có `-ncmoe`/`-fmoe`.
* **Gate:** chỉ giữ nếu ≥ +10% decode **trên chính model này**, cùng plan; nếu
  không thì ghi "đã đo, bỏ" như các mục âm ở `OPTIMIZE-tokps.md` §5.

### 3.7 Đã loại (giữ nguyên kết luận, đừng thử lại)

| Đòn bẩy | Vì sao loại |
|---|---|
| `--moe-cache-mib` (pool expert trong VRAM) trên card 4 GB | RFC [#24528](https://github.com/ggml-org/llama.cpp/discussions/24528) đo trên 4×3090 + 256 GB RAM mới +7…+57%; trên **1 GPU nhỏ** thì test độc lập cho **regression 19.32 → 13.25 t/s** và tác giả RFC xác nhận *"improvement is indeed quite hardware dependent"*. **Đo trên máy này: 1.172 → 0.472 t/s (−60%), load 58 → 140 s** (§4). Gate `_MOE_CACHE_MIN_SPARE_GB` giữ nguyên |
| `mlock` | Đã đo trên máy này: 6.26 vs 6.25 t/s (+0.2%), và `ulimit -l` = 8 MiB nên mọi đường mlock đều thành mmap thuần |
| `n_ubatch` > 512 | Đo trên chính gpt-oss-120b Q2_K: 512 → 6.42 t/s prefill, 2048 → 6.57 (phẳng), decode không đổi ⇒ 512 là knee |
| `--no-mmap` / O_DIRECT | Model 62.5 GB trên 15 GiB RAM; bỏ mmap nghĩa là đòi anonymous 62 GB ⇒ OOM, không phải nhanh hơn |
| KV f16 cho 64K | 10 GiB KV trên máy 15 GiB không swap — đã loại từ `OPTIMIZE-hermes4.md` |
| Tăng `-t` lên 8 | SMT contention; `-t 4` = số core vật lý đã đo là tối ưu |

---

## 4. Đã đo trên máy này — `tools/sweep_spill_moe.py`, 2026-10-28

`ctx 8192`, 32 token/lần, median 3 lần, **sau một request warm-up** (nên phép đo
không còn tính phần page-fault lần đầu — xem cảnh báo ở `docs/benchmarks.md`:
chỉ so số ở **cùng trạng thái cache**). Raw: `tools/spill_results/*.json`.

| Arm | gen t/s (median, cả 3 lần) | prefill t/s | đọc từ đĩa | VRAM | load |
|---|---|---:|---:|---:|---:|
| `classic` — `-ngl 2` (trước khi sửa) | **0.974** (0.974 / 0.888 / 1.005) | 11.35 | 0.827 GiB/token | 2932 MiB | 62 s |
| `attention` — `-ngl 37 -ncmoe 36` (sau khi sửa) | **1.172** (1.448 / 1.172 / 0.604) | **17.94** | 0.889 GiB/token | **1900 MiB** | 58 s |
| `attention` + `--moe-cache-mib 1024` | **0.472** (0.428 / 0.472 / 0.838) | 6.90 | 1.022 GiB/token | 2936 MiB | 140 s |

### 4.1 Kết luận từ số, không phải từ kỳ vọng

1. **Decode đang bị chặn bởi đọc đĩa, và bởi *kiểu* đọc chứ không phải băng thông.**
   0.83–1.02 GiB/token phải lấy từ storage, trong khi decode chỉ ~1 t/s ⇒ tốc độ
   đọc hiệu dụng **≈ 1 GiB/s**, so với **7.75–9.1 GB/s tuần tự** mà `bench/hwprobe.c`
   đã đo trên chính ổ này. Khoảng cách ~8× đó là **truy cập ngẫu nhiên từng expert
   slab 13.25 MB**, không phải ổ chậm. Đây là bằng chứng trực tiếp cho §3.3:
   prefetch/`madvise` biến fault ngẫu nhiên thành đọc lớn hơn, và đó là đòn bẩy
   **lớn hơn** đòn bẩy placement.
2. **`majflt/token` = 0.00 ở cả ba arm** — không phải vì không có page fault, mà vì
   readahead của kernel đã mang trang vào trước khi process fault (fault đó là
   *minor*). Vì vậy `read_gib_per_token` mới là số cơ chế đúng, `majflt` chỉ là phụ.
   Đây là điều bản thân harness đã học được trong lần chạy này (xem docstring).
3. **Placement mới: +20% decode (median), +58% prefill, và trả lại 1 GB VRAM.**
   Nhưng **phải đọc số một cách trung thực**: ba lần đo của arm `attention` trải từ
   **0.604 → 1.448 t/s**, tức biên độ lớn hơn mức chênh median so với `classic`.
   Hai thứ **ổn định** và đáng tin hơn: prefill 11.35 → 17.94 t/s và VRAM
   2932 → 1900 MiB (đúng như ngân sách: 1.39 GiB attention + 0.14 KV + compute).
   ⇒ Ghi vào `docs/benchmarks.md` **chỉ sau khi đo lại** với A/B chen kẽ (arm xen
   kẽ trong cùng phiên, như phương pháp ở `OPTIMIZE-tokps.md` §4). Chưa đủ để tuyên bố.
4. **Vì sao `attention` lại đọc đĩa *nhiều hơn* 0.06 GiB/token?** Vì `classic` giữ
   nguyên expert slab của 2 lớp trong VRAM (2 × 4 expert × 13.25 MB ≈ 0.10 GiB/token
   không phải đọc), còn `attention` đẩy cả 36 lớp expert về đĩa. Đổi lại, nó không
   còn đọc 1.39 GiB attention từ RAM mỗi token. Đây là **trao đổi có ý thức**, không
   phải phép màu — và nó giải thích luôn vì sao lợi ích ròng nhỏ hơn 42% lý thuyết.
5. **`--moe-cache-mib` là âm rõ rệt trên card 4 GB: −60% decode, load 140 s thay vì
   58 s.** Đúng như RFC #24528 cảnh báo và như test độc lập trên GTX 1080 Ti
   (19.32 → 13.25 t/s). Gate `_MOE_CACHE_MIN_SPARE_GB` của Vipera **giữ nguyên**;
   thí nghiệm này chỉ xác nhận gate, không mở lại.

### 4.2 Hạn chế của phép đo này

* Ba arm chạy **tuần tự**, mỗi arm load lại model ⇒ trạng thái page cache khác nhau
  giữa các arm. Đó là lý do chính khiến arm `attention` có một lần rơi xuống 0.604 t/s.
* Một máy, một model, một lần chạy median-3. Chưa phải số để trích dẫn ra ngoài.
* `prefill t/s` tính từ `prompt_tokens / ttft` như `vipera bench`, nên nó gồm cả chi
  phí cố định; chỉ so **giữa các arm**, không so với bảng A–E cũ.
* Prompt là prompt viết code ngắn, không phải prompt agent 64K. Ở 64K, KV và
  attention trên CPU sẽ đổi tỷ lệ — chưa đo.

### 4.3 Quant expert MXFP4 → `q2_0`: nhanh 3.7× nhưng phá model — **ĐÃ ĐO, KẾT LUẬN ÂM**

Thí nghiệm đầy đủ cho §3.2, chạy 2026-10-28. File gốc: 62.6 GB / 2 shard. File mới:
`~/.vipera/models/gpt-oss-120b-q2_0/gpt-oss-120b-q2_0-00001-of-00001.gguf`.

**Cấu hình quant** (chỉ đổi expert, phần còn lại *nâng* độ chính xác để cô lập biến):

```
llama-quantize --allow-requantize \
  --tensor-type "ffn_.*_exps\.weight=q2_0" \
  <shard1> <out> q8_0 4      # 4 = nthreads, positional
```

| | file gốc | file `q2_0` |
|---|---:|---:|
| kích thước | 59 659 MiB (4.28 BPW) | **33 101 MiB (2.38 BPW)** |
| expert | MXFP4 4.25 bpw, 108 tensor | `q2_0` 2.06 bpw, 108/108 tensor |
| attention + embedding | q5_0 / q8_0 | q8_0 (145 tensor) |
| thời gian quant | — | **268 s** (4 core), 0 lỗi |
| fallback im lặng | 217 (khi không override) | **0** |

**Chất lượng — hai phép đo độc lập, cùng một protocol:**

| | PPL (wikitext-2 test, 4×512, `-ngl 37 -ncmoe 36 -nr`) | sinh văn bản (`"The capital of France is"`, temp 0) |
|---|---:|---|
| file gốc | **410.96** ± 36.8 | `Paris.\n\nThe capital city of France is Paris.` — đúng, chỉ lặp |
| `q2_0` | **879.53** ± 86.6 (**×2.14**, +0.76 nats/token) | `also known as a "c" ... I apologize for the confusion` — **rác** |

**Tốc độ — 3 lần/arm, server riêng cho từng arm, cùng cờ (`-c 4096 -ngl 37 -ncmoe 36 -nr`), 32 token:**

| arm | từng lần (t/s) | **median** | prefill (t/s) |
|---|---|---:|---:|
| file gốc | 1.238 / 0.874 / 0.407 | **0.874** | 0.67 / 0.89 / 0.35 |
| `q2_0` | 3.204 / 2.992 / 3.196 | **3.196** | 1.94 / 2.25 / 3.50 |

**Đọc số cho đúng:**

1. **Nhanh hơn thật: ×3.7 median.** Nhưng không phải ×1.8 như "byte/token giảm một
   nửa" gợi ý. Hai lý do: (a) expert là 97.7% byte của model nên giảm 44% byte chỉ
   còn ~1.9× lý thuyết; (b) phần còn lại của decode không tỷ lệ với byte đĩa.
2. **Độ ổn định mới là thứ đáng giá hơn con số trung bình.** Arm gốc **xấu dần
   đều** 1.238 → 0.874 → 0.407 trong cùng một phiên (page cache bị đẩy ra liên tục
   vì 62.6 GB trên 15 GB RAM), còn `q2_0` gần như phẳng 2.99–3.20 vì 34.7 GB nằm
   gọn hơn nhiều trong cache. Đây đúng là cơ chế ở §4.1: vấn đề là *trạng thái
   cache*, không chỉ băng thông.
3. **Nhưng chất lượng thì mất hết, và mất theo kiểu không cứu được bằng flag.**
   PPL ×2.14 kèm văn bản rác = model đã hỏng, không phải "kém hơn một chút".
   Đúng như cảnh báo published ở §3.2 ("quality collapse is steep and non-uniform"),
   và đúng như dự đoán từ việc `q2_0` là **RTN đối xứng 2 bit, không imatrix**:
   expert weight của gpt-oss có outlier lớn, không có min/zero-point thì scale bị
   outlier chi phối.
4. **Prefill hầu như không cải thiện** trong ppl: 530.60 s/pass (62.6 GB) →
   494.05 s/pass (34.7 GB), chỉ **−7%** dù byte giảm 44%. Vì một pass ppl là batch
   512 token: chi phí tính toán mỗi token (4 core, expert trên CPU) chứ không phải
   I/O mới là chỗ chặn. Đừng kỳ vọng quant giúp prefill trên máy này.
5. **Kết luận:** đòn bẩy "nén expert" **đóng** cho model này, không phải vì ý tưởng
   sai mà vì (a) hình học 2880 chặn mọi type 2-bit chất lượng cao, (b) type duy nhất
   vừa hình học thì bỏ qua imatrix. Muốn giữ chất lượng thì phải giảm byte bằng cách
   khác: §3.3 (prefetch, không đổi trọng số) và §3.5 (prune expert, bỏ hẳn expert ít
   dùng thay vì làm nhễu tất cả).

**Ghi chú vận hành (đã đụng trong lúc đo):**

* Trên box 15 GB RAM, `llama-perplexity` **OOM ngay khi load** file 62.6 GB:
  `failed to allocate CPU_REPACK buffer of size 61182812160`. Phải thêm **`-nr`**
  (`--no-repack`). File 34.7 GB thì không cần, nhưng cứ dùng `-nr` cho cả hai arm để
  so sánh cùng điều kiện.
* `llama-quantize` nhận thread count là **positional cuối**, không phải `--threads`
  (truyền flag sẽ chỉ in usage rồi thoát, trông như "không làm gì").
* Chạy job dài phải `exec` trong tiến trình nền được quản lý; `nohup ... &` trong
  một lệnh đồng bộ bị dọn theo shell — quantize đã chết một lần ở tensor 91 vì vậy
  (không phải OOM), và `llama-server` cũng "ready" rồi biến mất một lần.


---

## 5. Harness — `tools/sweep_spill_moe.py`

```
python tools/sweep_spill_moe.py --print-arms          # xem argv của từng arm, không load model
python tools/sweep_spill_moe.py --arms attention      # đo một arm
python tools/sweep_spill_moe.py --tokens 32 --repeats 3
```

Mỗi arm chạy **chính `Plan` của Vipera** qua `backends.server.server_command()`,
nên cờ được đo đúng là cờ mà gateway sẽ dùng. Harness ghi, cho mỗi arm:

* `gen_tps_median` (median 3 request), `prefill_tps_median`, TTFT từng lần;
* **`read_gib_per_token`** và **`major_faults_per_token`** — lấy từ `/proc/<pid>/io`
  trong đúng cửa sổ đo. Đây là số quyết định: hai arm cùng tok/s mà khác majflt/token
  là hai cơ chế khác nhau, và arm nào ít majflt hơn thì mới **thật sự** bớt phụ thuộc đĩa;
* `peak_rss_gib`, `vram_used_mib`, `load_s`;
* argv đầy đủ + đường dẫn log của llama-server.

Kết quả: `tools/spill_results/<alias>-<timestamp>.json` (JSON thô, copy một dòng là
chạy lại được).

---

## 6. Thứ tự làm tiếp

```
0. ~~§3.2 quant lại expert (IQ2/Q2_K + imatrix)~~               ← ĐÓNG 2026-10-28 (§4.3):
   hình học 2880 chặn mọi type 256-block; q2_0 vừa hình học thì bỏ qua imatrix,
   nhanh 3.7× nhưng PPL ×2.14 + văn bản rác. Đừng thử lại.
1. §3.3 build llama-server CUDA có prefetch theo routing     ← giờ là kỳ vọng lớn nhất
   (nvcc đã có; patch đã viết ở ~/dev/vipera/bench) + đo majflt/token trước/sau
2. §3.4 bật --spec-type ngram-map-k cho ca code/rewrite      ← chỉ cần config + đo
3. §3.6 build ik_llama.cpp, so cùng plan                     ← published 1.9×, phải tự đo
4. §3.5 prune expert                                         ← cần permute router
   (giờ là con đường *duy nhất* còn lại để giảm byte/token — vì §3.2 đã đóng)
```

**Gate chung** (không đổi): một mục chỉ được ghi vào README/benchmarks khi có
(a) JSON thô trong repo, (b) lệnh copy-paste lại được, (c) Δ > 5% so với control
hoặc lý do chất lượng (PPL/KL) — và **không đổi ngưỡng sau khi đã thấy số**.

---

## 7. Nguồn

| Nguồn | Dùng cho |
|---|---|
| [Performant local MoE CPU inference with GPU acceleration in llama.cpp](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) | Nguyên tắc "always-active lên GPU", `-ot exps=CPU` ≡ `-cmoe`, op-offload prompt processing, `-b/-ub` lớn cho CPU+GPU |
| [llama.cpp RFC #24528 — MoE expert cache](https://github.com/ggml-org/llama.cpp/discussions/24528) | Thiết kế cache expert trong VRAM, số +7…+57% trên máy lớn, **regression trên 1 GPU nhỏ (19.32 → 13.25 t/s)**, locality ~top 10% expert ≈ 80% hit |
| [llama-cpp-expert-sniper](https://huggingface.co/waltgrace/llama-cpp-expert-sniper) | madvise prefetch theo top-k của router: 0 → 0.57 t/s khi model > RAM; LRU cache thua madvise; chỉ có lợi khi RAM < model |
| [llama.cpp issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480) | ik_llama.cpp ~1.9× cho MoE inference (fused MoE, AVX-512) |
| [modelfit.io — GPT-OSS 120B quant compare](https://modelfit.io/quant-compare/gpt-oss-120b/) | Cảnh báo chất lượng khi quant ≤ 2 bpw |
| `OPTIMIZE-tokps.md`, `OPTIMIZE-techniques.md`, `docs/benchmarks.md` (repo này) | Số đo cũ trên máy này: threads, reserve VRAM, placement ladder, ngram +37%, ub 512 là knee, các cực âm |
| `~/dev/vipera/bench/*.patch`, `docs/REWRITE.md`, `bench/results/prune-ppl-indomain.json` | Bảng prefetch theo routing, prune expert, và tại sao reorder phải kèm permute router |
