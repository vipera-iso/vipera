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
| **T1** | **MoE expert offload** (`-ot exps=CPU`, `--cpu-moe`, `-ncmoe N`) | **Có trong `llama-server`** (0.7.0-dev), **không** có trong `llama-cpp-python` | LFM2.5-8B-A1B: **4.35 GB expert / 4.80 GB tổng = 91%** | Rất cao về lý thuyết | **Đã đo: hoà** — `-ngl 24 -ncmoe 12` 69.16 t/s vs layer split `-ngl 16` 68.99 t/s; prefill 206 vs 211 t/s. Không có lợi trên card 4 GB (§3) |
| **T2** | Spec decoding không cần draft model (`--spec-type ngram-*`) | Có trong `llama-server` | Prompt code/rewrite lặp lại nhiều | +10–30% nếu acceptance cao | **Đã đo: THẮNG** — `ngram-map-k` 8.54 vs 6.23 t/s (**+37%**) trên prompt rewrite file; prose −1%. Đã bật mặc định qua `spec_type` (§4) |
| **T3** | `--cache-reuse N` (KV shifting cho prefix trùng) | Có trong `llama-server`, mặc định 0 (tắt) | Agent loop = cùng system prompt | Giảm TTFT | **Đã đo: nhiễu** — TTFT 0.85 s vs 0.87 s; prompt cache của server đã phủ đúng ca prefix trùng |
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
| **T14** | Spec decoding với draft model (`-md`) / prompt lookup trong binding | Có (`draft_model=` trong llama-cpp-python) | DFlash cần ≥6 GB VRAM; `draft-simple` cần model cùng vocab | +25% nếu acceptance tốt | **Đã đo: THUA** — `LlamaPromptLookupDecoding` 4.27/4.14 vs 6.05 t/s (**−29/−32%**): việc verify chạy ở Python, mỗi bước vẫn một lần decode đầy đủ. Đã gỡ khỏi code (§5)

---

## 3. T1 — MoE expert offload: đã đo, hoà với layer split

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
["split_mode", "tensor_split", "use_mmap", "offload_kqv", "op_offload", "numa", "draft_model"]
# không có override_tensor / cpu_moe / n_cpu_moe
```

⇒ Muốn dùng T1 **bắt buộc** `backend: "server"`. `server_command()` hiện chỉ phát
`-c -ngl -t -b -ub -fa --jinja -ctk -ctv -lm`, **không** phát `-ot/-cmoe/-ncmoe`
(đúng như known-gap trong `docs/architecture.md`).

**Đã đo (2026-10-23) bằng `llama-bench` của build CUDA, LFM2.5-8B-A1B-Q4_K_M
(4.79 GiB, 24 layer, 32 expert, top-4), 4 thread, `-r 2`:**

| arm | argv | decode tg96 | prefill pp512 |
|---|---|---|---|
| CPU-only | `-ngl 0` | 37.22 t/s | 177.13 t/s |
| layer split (hiện tại) | `-ngl 16` | **68.99 t/s** | 211.34 t/s |
| expert offload, cân bằng | `-ngl 24 -ncmoe 12` | **69.16 t/s** | 206.11 t/s |
| expert offload, nhiều expert ở RAM | `-ngl 24 -ncmoe 16` | 60.27 t/s | — |
| expert offload, hết expert ở RAM | `-ngl 24 -ncmoe 20` | 52.96 t/s | — |
| chỉ expert ở RAM | `-ngl 99 -ot exps=CPU` | 47.47 t/s | — |
| split + expert ở RAM | `-ngl 16 -ncmoe 99` | 44.49 t/s | — |

**Kết luận: bỏ.** Trần của MoE offload trên card này là **69.16 t/s**, tức bằng
đúng layer split 16 layer (68.99, Δ 0.2% trong nhiễu). Lý do không phải cấu hình
sai mà là số học: model chỉ 4.79 GiB nên 16 layer **đã** mang theo cả expert của
nó lên VRAM; `-ncmoe` chỉ đổi *thứ tự* cái gì nằm ở đâu, không tăng được phần
nào của model được tính trên GPU. Expert chạy trên CPU 4 thread chậm hơn nhiều
so với chạy trên GPU, nên đẩy nhiều expert xuống RAM là mất (20 → 53 t/s).
Trường hợp `-ot exps=CPU` kiểu gpt-oss-120B trong tài liệu chỉ thắng khi VRAM
quá nhỏ so với model — ở đây không có.

Ghi chú vận hành giữ lại: `-ngl 24` với `-ncmoe 8/12` **load được** ở chỗ
`-ngl 24` thuần **OOM** (nó cần 3.68 GB trước khi tạo context) → planner vẫn
giữ ladder `-ncmoe` làm đường dự phòng, nhưng nó không phải đường nhanh hơn.
Đo lại bằng: `tools/measure_knobs.py` (in-process) hoặc `llama-bench -m <model>
-ngl 24 -ncmoe 12 -p 0 -n 96` / `-p 512 -n 0` (server).

---

## 4. T2/T3 — speculative + cache-reuse: T2 thắng, T3 không đổi gì

Đo bằng `llama-server` build CUDA (b11827), Q3-14B, 8K, `q8_0`, `-ngl 17`,
`-fa on`, 128 token, 3 lần/arm, prompt = file Python cần rewrite kèm type hint
(ca "rewrite code" thật của Vipera):

| arm | decode t/s | runs |
|---|---|---|
| control | 6.227 | 6.32 / 6.23 / 6.17 |
| `--spec-type ngram-simple` | 8.115 | 8.10 / 8.11 / 8.14 |
| `--spec-type ngram-map-k` | **8.540** | 8.12 / 8.63 / 8.54 |
| prose, control | 6.398 | 6.43 / 6.40 / 6.38 |
| prose, `ngram-map-k` | 6.329 | 6.31 / 6.33 / 6.33 |

⇒ **+37% khi câu trả lời lặp lại nội dung prompt** (rewrite/diff/patch — đúng ca
agent), **−1% trên prose mới** (nằm trong nhiễu: drafter chỉ tốn một lần hash
lookup mỗi bước). Lợi ích co theo **tỉ lệ chữ được copy từ prompt** và **độ dài
câu trả lời**: cùng prompt đó nhưng chỉ sinh 48 token → 6.48 t/s (không lợi),
64 token trên prompt snippet ngắn → 6.45 vs 6.52 (không lợi). Nói cách khác:
nó thắng ở ca "viết lại cả file" (≥128 token, phần lớn là copy), không thắng ở
ca trả lời ngắn.
Raw: `tools/server_results/ngram_prose_cache.jsonl`, harness `tools/server_ab.py`. Vì vậy nó được bật mặc định (`spec_type: ngram-map-k`) cho
backend `server`, và **chỉ** ở đó: llama-cpp-python không có `--spec-type`, còn
prompt-lookup của binding thì đo được **−29/−32%** (xem §5) nên đã bị gỡ khỏi
`engine.py`.

T3 `--cache-reuse 256` (turn 2 = prompt cũ + một câu nối tiếp): TTFT 0.85 s vs 0.87 s
khi tắt — không có gì để lấy, vì prompt cache của server đã xử lý đúng ca
prefix trùng nguyên vẹn (TTFT 0.15–0.18 s khi prompt lặp hoàn toàn).

Cả hai đều **miễn phí** về VRAM nên không có rủi ro OOM.

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

**Prompt lookup ngay trong process** (cùng prompt code, `tools/measure_knobs.py`,
Q3-14B @17 layer, 3 lần/arm):

| arm | decode t/s | TTFT |
|---|---|---|
| control | 6.047 | 5.59 s |
| ngram 2, 10 token dự đoán | 4.268 | 6.05 s |
| ngram 3, 8 token dự đoán | 4.138 | 6.04 s |

⇒ **−29% / −32%.** Đã xoá `spec_decoder` + tham số `draft_model` khỏi
`engine.py`; muốn spec decoding thì dùng backend `server` với `--spec-type`
(§4), nơi vòng verify chạy bằng C++.

**Quant IQ3 (R2 trong `OPTIMIZE-tokps.md`)** — `llama-quantize` từ Q8_0 +
imatrix, cùng `q8_0` KV, 8K, 96 token:

| quant | ngl | size | decode t/s |
|---|---|---|---|
| Q3_K_M (baseline) | 17 | 6.82 GiB | **6.229** |
| IQ3_XXS | 17 | 5.53 GiB | 3.434 |
| IQ3_XXS | 21 | 5.53 GiB | 3.533 |

⇒ **−45%**, dù nhỏ hơn 1.3 GiB. Trên sm_75 không có kernel MMQ cho IQ3, nên
iq3 lấy lại đúng phần đã mất về bandwidth. Gate "ΔPPL ≤ 0.4 và decode ≥ +8%"
chết ngay ở vế thứ hai, nên PPL **không cần chạy** và không được trích như số
đo. `IQ3_S` không dùng được: ladder `-ngl` của nó OOM giữa sweep (CUDA error
trong `ggml_cuda_pool_vmm::alloc`).

---

## 6. Thứ tự làm tiếp (thay đổi so với `OPTIMIZE-tokps.md`)

```
1. T2  ngram spec cho backend server        ← ĐÃ LÀM: spec_type=ngram-map-k, +37% rewrite
2. T1  MoE offload cho LFM2.5               ← đã đo, hoà: không làm thêm
3. T3  --cache-reuse cho agent loop         ← đã đo, nhiễu: không làm
4. P3.2 tách lane 64K (hybrid)              ← giữ nguyên kết luận cũ
Không làm: T8–T13 (xem lý do ở §2)
```

**Blocker cũ đã gỡ, không cần biên dịch.** Build CUDA tại chỗ thất bại cả hai
đường: `nvcc 12.9` từ chối `gcc 16.2` (host compiler không hỗ trợ) và `clang 23`
làm `cudafe++` chết ở `crt/math_functions.h` vì glibc mới thêm `noexcept`. Thay
vào đó dùng **binary dựng sẵn** `llama-b11827-bin-ubuntu-cuda-12.8-x64.tar.gz`
(ggml-org/llama.cpp releases) + `libcudart` 12.9 giải nén vào `~/cuda/lib`:
`ldd` sạch, `llama-cli --list-devices` thấy `CUDA0: GTX 1650 (3674 MiB free)`,
`llama-server --version` → `build 11827`. Mọi số T1/T2/T3 ở trên đo bằng binary
này, không phải bằng build local.

**Còn lại (không phải blocker, chỉ là chưa cần):** T14 (draft model `-md`, cần
tải thêm Qwen3-0.6B) — ngram đã cho +37% với chi phí 0, nên draft model chỉ
đáng thử nếu cần thêm trên prose.

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
