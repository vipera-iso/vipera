# Nghiên cứu tài liệu tối ưu — 2026-10-23

> Mục đích: đọc hết tài liệu tối ưu đang có (trong repo và pack ngoài), rồi trả lời
> đúng một câu: **kỹ thuật nào còn dùng được trên máy này, kỹ thuật nào không, và
> cái nào cần đo trước khi tin**.
>
> Quy tắc giữ nguyên từ `OPTIMIZE-tokps.md`: số đo thắng văn bản; mỗi mục chỉ
> được ghi vào README/benchmarks khi có (a) raw run trong `tools/*_results/`,
> (b) lệnh chạy lại được, (c) Δ > 5% hoặc lý do chất lượng.

---

## 1. Đã đọc gì

| Nguồn | Nội dung dùng được |
|---|---|
| `docs/references/stack/llama-cpp-speculative.md` | `--spec-type` (draft-simple/eagle3/dflash/dspark/mtp, `ngram-*`), tham số `--spec-draft-n-max`, thống kê acceptance |
| `docs/references/stack/llama-cpp-server-README.md` | Toàn bộ flag server: `-ot/--override-tensor`, `-cmoe/--cpu-moe`, `-ncmoe`, `-ncffn`, `--op-offload/--no-op-offload`, `-lm` (`mlock`, `dio`), `--cache-reuse`, `-cb`, `-np`, `-kvu`, `-fa` |
| `docs/references/stack/llama-cpp-python-README.md` + API reference | Tham số `Llama(...)` thật của binding (đối chiếu §4) |
| `docs/references/papers/*.pdf` | FlashAttention, KIVI, TurboQuant/PolarQuant, PagedAttention, spec decoding, DFlash, StreamingLLM, GPTQ |
| `~/Downloads/vipera-tai-lieu-co-gia-tri/S/hf-blog-llamacpp-moe-offload-guide.md` | Công thức MoE offload: `-ngl 999 -ot "exps=CPU"` / `--n-cpu-moe N`, `-b/-ub` cho prompt processing, `GGML_OP_OFFLOAD_MIN_BATCH`, `--no-mmap` |
| `…/S/hardware-corner-gpt-oss-moe-offloading.md` | Số đo thật của `--n-cpu-moe` trên RTX 3090 với gpt-oss-120B |
| `…/S/github-discussion-15280.md` | Thực chiến: MoE PP chậm, `--cpu-moe` + `--no-op-offload`, thứ tự tăng `--ubatch-size` |
| `…/S-mo-rong/INDEX.md` + PDFs | Pack open-problems: DALI (MoE offload trên PC), KVSwap/Tutti (KV ra SSD), Dovetail/SpecOffload/SPARQ (spec trong chế độ offload), cheops (I/O model) |
| `…/A/` (AWQ, H2O, StreamingLLM, FineMoE), `…/B/` (QMoE, HAWQ, MxMoE, MoPEQ, Atom) | Kỹ thuật cấp thuật toán (quant theo kênh, evict KV, expert placement) |

---

## 2. Bảng xếp hạng — cái gì còn cửa

| # | Kỹ thuật | llama.cpp / binding có? | Trên máy này | Kỳ vọng | Trạng thái |
|---|---|---|---|---|---|
| **T1** | **MoE expert offload** (`-ot exps=CPU`, `--cpu-moe`, `-ncmoe N`) | **Có trong `llama-server`** (0.7.0-dev), **không** có trong `llama-cpp-python` | LFM2.5-8B-A1B: **4.35 GB expert / 4.80 GB tổng = 91%** | Rất cao cho lane MoE: giữ attention+KV trên 4 GB VRAM, expert ở RAM | **Chưa đo được** (llama-server local build CUDA=OFF) — cần P2.1 |
| **T2** | Spec decoding không cần draft model (`--spec-type ngram-*`, `--spec-default`) | Có trong `llama-server` | Agent prompt lặp prefix nhiều | +10–30% nếu acceptance cao | Chờ T1's build (cùng một `llama-server` CUDA) |
| **T3** | `--cache-reuse N` (KV shifting cho prefix trùng) | Có trong `llama-server`, mặc định 0 (tắt) | Agent loop = cùng system prompt | Giảm TTFT, không đổi decode | Chỉ plumbing; đo sau khi có server CUDA |
| **T4** | `-b/-ub` lớn cho prompt processing (op-offload ≥32 token) | Có | Đã đo: `ub 1024/2048` **không load nổi** ở 17–14 layer; cùng offload thì TTFT bằng nhau | 0 trên card 4 GB | **Đóng (âm tính)** |
| **T5** | `--no-op-offload` (khuyến nghị khi bandwidth-limited) | Có, cả binding (`op_offload=False`) | Đã đo: decode −1.7% (nhiễu), **TTFT +18%** | 0 / hại | **Đóng (âm tính)** |
| **T6** | `--no-mmap` (guide khuyến nghị) | Có, cả binding (`use_mmap=False`) | Đã đo: decode −1.6%, TTFT −2.4% | 0 | **Đóng (âm tính)** |
| **T7** | `mlock` / `-lm mlock` | Có | Đã đo với `RLIMIT_MEMLOCK=unlimited`: +0.2% | 0 | **Đóng (âm tính)** |
| **T8** | `ik_llama.cpp` (`--merge-qkv`, `-gr`, `-mla`, `-sm graph`) | **Không** — chỉ có trong fork | — | Có thể +5–15% | **Không làm**: sẽ phải thay runtime, binding hiện tại là mainline |
| **T9** | KV < 4 bit (TurboQuant/PolarQuant, KIVI per-channel K) | **Không** — llama.cpp dừng ở `q4_0`/`iq4_nl` | 64K vẫn là tường | Cao về lý thuyết | **Không làm**: là code C++/kernel mới, không phải flag |
| **T10** | StreamingLLM / H2O / KV-evict (attention sink) | **Không** — llama.cpp không có sliding-window KV | — | Cao cho 64K | **Không làm** (giữ kết luận cũ của `OPTIMIZE-tokps.md`) |
| **T11** | KV ra SSD (KVSwap, Tutti) | **Không** | Model đã vừa RAM (`disk_mib_per_token = 0`) | 0 cho ca của mình | **Không làm** |
| **T12** | Expert prefetch / surrogate (DALI, SpecPrefetch, SPICE) | **Không** | Cần trace + loader riêng | Cao cho MoE spill | **Không làm** (research; chỉ mở lại nếu T1 chứng minh expert ở RAM là nút thắt) |
| **T13** | PagedAttention / continuous batching | `llama-server` có (`-cb`, `-np`, `-kvu`) | Vipera serialize 1 request bằng lock | **0** cho 1 user | **Không làm** |
| **T14** | Spec decoding với draft model (`-md`) | Có | DFlash cần ≥6 GB VRAM; `draft-simple` cần model cùng vocab | +25% nếu acceptance tốt | Chờ T2: ngram rẻ hơn, thử trước |

---

## 3. T1 — MoE expert offload: cái cửa lớn nhất còn lại (và cách đo)

**Vì sao đáng làm.** Trên LFM2.5-8B-A1B (`tools/gguf_analyzer.py` đọc index):

```
expert tensors : 88  →  4.35 GB
toàn bộ file   :        4.80 GB   (91% là routed expert)
attention + ffn + norm + embed : ~0.45 GB
```

Planner hiện chia layer đều: 24 layer × 0.2 GB → ~17 layer lên GPU, tức là **mỗi
layer đem theo cả attention lẫn expert của nó**. Cách llama.cpp khuyến nghị cho
MoE thì ngược lại: `-ngl 999 -ot "exps=CPU"` (hoặc `--n-cpu-moe N`) → **toàn bộ
attention + KV lên VRAM (chỉ ~0.45 GB + KV), chỉ expert ở RAM**. Với 4 GB VRAM,
đây là khác biệt giữa "17/24 layer, mỗi token đọc 0.2 GB từ RAM" và "24/24 layer
attention trên GPU, mỗi token chỉ đọc expert được chọn".

Nguồn: HF blog (công thức + `GGML_OP_OFFLOAD_MIN_BATCH`), Hardware Corner (đo thật
gpt-oss-120B: layer split 0.90 t/s → `--n-cpu-moe`), discussion 15280 (`--cpu-moe`
+ `--no-op-offload` khi bandwidth-limited).

**Chặn kỹ thuật (đã kiểm, không phải phỏng đoán).**

```python
# llama_cpp 0.3.36 — các tham số liên quan:
['split_mode', 'tensor_split', 'use_mmap', 'offload_kqv', 'op_offload', 'numa', 'draft_model']
# không có override_tensor / cpu_moe / n_cpu_moe
```

⇒ Muốn dùng T1 **bắt buộc** `backend: "server"`. `server_command()` hiện chỉ phát
`-c -ngl -t -b -ub -fa --jinja -ctk -ctv -lm`, **không** phát `-ot/-cmoe/-ncmoe`
(đúng như known-gap trong `docs/architecture.md`).

**Việc cần làm (chưa làm, chờ đo được).**

1. `planner`: thêm `Plan.moe_experts_on_cpu: bool` + `Plan.expert_share: float`,
   suy từ meta (`expert_bytes / size_bytes`) khi GGUF index có `category_sizes`.
   Ngưỡng đề xuất: expert ≥ 60% file **và** `vram_free` không đủ cho cả model.
2. `server_backend.server_command()`: nếu `moe_experts_on_cpu` → `-ngl 999
   -ot "exps=CPU"`, rồi nếu còn VRAM thì `-ncmoe N` (N giảm dần) thay vì layer
   split; ghi rõ trong argv test.
3. Test argv + một smoke test `llama-server` CPU-only (chạy được hôm nay, chứng
   minh flag hợp lệ), **số t/s thì phải chờ build CUDA**.

**Gate để chấp nhận:** trên LFM2.5 (8K, q8_0), so `-ngl 17` (hiện tại) với
`-ngl 999 -ot exps=CPU`, cùng `llama-server` build CUDA: TG ≥ +20% và PP không
giảm > 10%. Nếu không đạt → ghi "đã đo, bỏ" và giữ layer split.

---

## 4. T2/T3 — speculative + cache-reuse: đường miễn phí, cùng một blocker

- `--spec-type ngram-*` không cần model draft, không tốn VRAM (`ngram-mod` ăn
  ~16 MB). Điều kiện ăn: văn bản lặp lại (agent, code, rewrite) — đúng ca
  Vipera phục vụ. Paper `speculative-decoding-2211.17192` và `dflash` giải thích
  vì sao spec chỉ có lời khi **bytes là nút thắt** — với 26/40 layer nằm ở RAM,
  đây chính là chế độ đó (Dovetail/SpecOffload đo trên target CPU-resident).
- `--cache-reuse 256` bật KV shifting cho prefix trùng; in-process đã tự reuse
  prefix, `llama-server` thì mặc định tắt (0).
- Cả hai đều là flag của `llama-server` ⇒ **cùng blocker với T1**: build CUDA.

---

## 5. Những gì đã đo trong phiên này và bị loại

Interleaved A/B/C cùng process, Q3-14B, 8K, `q8_0`, 17/40 layer, 3 lần mỗi arm
(`tools/measure_knobs.py`):

| arm | decode t/s | TTFT | verdict |
|---|---|---|---|
| control | 5.563 (5.563/5.466/5.616) | 2.509 s | — |
| `op_offload=False` | 5.469 | 2.963 s | **bỏ** (decode nhiễu, TTFT +18%) |
| `use_mmap=False` | 5.474 | 2.450 s | **bỏ** (Δ trong nhiễu, load chậm hơn) |

Cộng với các kết quả âm tính đã có: `n_ubatch` 256/1024/2048 (§G8), `mlock` với
`RLIMIT_MEMLOCK` đã nâng (§G5), `n_threads_batch` (§G2).

---

## 6. Thứ tự làm tiếp (thay đổi so với `OPTIMIZE-tokps.md`)

```
1. P2.1  build llama-server với CUDA           ← mở khoá T1, T2, T3 cùng lúc
2. T1    MoE offload cho LFM2.5 (lane agent)   ← kỳ vọng lớn nhất
3. T2    ngram spec cho Q3-14B (8K)            ← +10-30% nếu acceptance tốt
4. T3    --cache-reuse cho agent loop          ← giảm TTFT
5. P3.2  tách lane 64K (hybrid) — giữ nguyên kết luận cũ
Không làm: T8–T13 (xem lý do ở §2)
```

**Blocker duy nhất của 1–4:** máy không có `nvcc` (`which nvcc` → not found,
`~/dev/llama.cpp/build/CMakeCache.txt` → `GGML_CUDA:BOOL=OFF`). Cần cài CUDA
toolkit (~3 GB) hoặc dùng build CUDA dựng sẵn; mọi kết luận về T1/T2/T3 **chưa
được đo trên máy này** và không được trích như số thật.

---

## 7. Điều tài liệu nói mà **không** áp dụng cho Vipera

- `-ot` / `--n-cpu-moe` cho **dense** 14B: không có tensor expert ⇒ vô nghĩa
  (`--n-cpu-ffn` là biến thể cho dense, chưa ai đo; mở lại chỉ khi có MoE kết quả).
- NUMA (`--numa distribute`, numactl): máy 1 socket ⇒ không có tác dụng.
- Multi-GPU (`-sm graph/row/tensor`, NCCL): 1 GPU.
- `--parallel`, `-kvu`, continuous batching: Vipera serialize mọi request bằng
  lock (1 user), nên không phải nút thắt tốc độ.
- PagedAttention/vLLM: giải bài toán *nhiều request*, không phải tok/s của 1
  request.
