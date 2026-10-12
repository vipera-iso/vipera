# ALGORITHMS — Phân tích thuật toán chạy LLM/MoE trên thiết bị yếu

> Tài liệu này là **phần quan trọng nhất** của dự án (mục 1 trong prompt gốc).
> Mọi con số in đậm là **đo trên máy mục tiêu** hoặc **suy ra từ đo**; con số khác ghi rõ là ước lượng.
> Các kết quả của FreeToken/AirLLM/edge0 là **theo công bố của chính họ**, chưa kiểm chứng lại ở đây.

---

## 0. Máy mục tiêu (đo 12/10)

| Thành phần | Thông số | Ý nghĩa với thuật toán |
|---|---|---|
| CPU | Intel i5-11300H, **4 core / 8 thread**, tối đa 4.4 GHz | tính toán MoE rơi vào CPU sẽ chậm; bandwidth DDR4 mới là trần |
| RAM | **15 GiB**, khả dụng ~10 GiB, **không swap** | model phải ≤ ~11 GiB nếu muốn resident; quá ngưỡng ⇒ phải streaming |
| GPU | **GTX 1650 Max-Q, 4 GB VRAM**, compute 7.5 (Turing) | chứa attention + KV + vài expert; **không chứa nổi cả model** |
| iGPU | Intel Iris Xe (TigerLake) | có thể tính chung nhưng không có API hợp nhất với NVIDIA |
| Disk | **WD SN530 NVMe (DRAM-less)**, ~404 GB trống | streaming đọc tuần tự tốt, nhưng **DRAM-less** ⇒ burst/random kém hơn NVMe có DRAM |
| Có sẵn | `python3`, `cmake`, `gcc` | thiếu `llama.cpp`, `ollama`, `cargo`, `nvcc` |

**Trần lý thuyết (suy ra, không phải đo):** decode LLM là bài toán **memory-bandwidth-bound**.
- RAM DDR4-3200 dual-channel ≈ **40–45 GB/s** thực dụng.
- VRAM GTX 1650 ≈ **192 GB/s**.
- NVMe SN530 đọc tuần tự ~**1.5–2.0 GB/s** thực dụng (DRAM-less).

⇒ Mỗi token phải đọc bộ trọng số *đang hoạt động*. Vậy:
`tokens/s ≈ bandwidth / (bytes trọng số hoạt động mỗi token)`.

---

## 1.a Layer-wise Streaming (AirLLM)

**Cơ chế.** Không nạp cả model; nạp **một layer** vào compute unit, tính, giải phóng, sang layer kế.

```
VRAM_đỉnh = max(kích thước 1 layer) + KV + activation + overhead   (KHÔNG phụ thuộc tổng model)
```

**Trade-off trung tâm.** Đổi *dung lượng* lấy *I/O*. Với model **lớn hơn RAM**, streaming là bắt buộc.
Với model **vừa RAM**, streaming là tự sát về tốc độ.

**Con số trên máy này.** Một MoE 30B Q4 (≈18.6 GB) không vừa 11 GiB RAM ⇒ phải streaming từ NVMe:
```
18.6 GB / 2.0 GB/s ≈ 9.3 s cho MỘT lượt đọc toàn bộ
```
Nếu mỗi token phải đọc lại toàn bộ ⇒ **~0.1 token/s** — vô dụng. ⇒ Streaming chỉ thắng khi
**phần lớn trọng số nằm resident**, chỉ stream phần ít dùng.

**Tối ưu I/O (thứ tự ưu tiên):**
1. **mmap** trọng số (`mmap` + `MADV_WILLNEED`) để OS page cache lo việc đọc.
2. **Prefetch layer N+1 trong lúc tính layer N** (một luồng I/O riêng + ring buffer).
3. **Double buffering**: 2 buffer, DMA/đọc async trong khi compute buffer kia.
4. **Pinned (page-locked) host memory** để H2D qua PCIe không bị chặn.
5. **NVMe là trần cứng** — không thuật toán nào vượt được 2 GB/s đọc đĩa.

**Áp dụng cho dense vs MoE:** dense buộc stream *cả layer*; MoE có thể stream **theo expert**
(hạt mịn hơn ⇒ ít I/O hơn nhiều). Đây là lý do 1.c/1.b quan trọng hơn 1.a trên máy yếu.

---

## 1.b MoE Inference trên thiết bị yếu

**Cơ chế.** MoE có `N` expert/layer, mỗi token chỉ đi qua `k` expert (top-k). Chỉ nạp `k` expert đó.

```
VRAM/RAM_cần = Σ (shared/attention, resident) + max(k expert) + router   (nhỏ hơn nhiều so với Σ N expert)
```

| Model | Tổng | Active/token | Lớp | Expert/lớp | top-k | Q4_K_M (ước lượng) |
|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 15.7B | 2.4B | 27 | 64 + 2 shared | 6 | **~9.5–10.4 GB → VỪA 11 GiB RAM** |
| Qwen3-30B-A3B | 30.5B | 3.3B | 48 | 128 | 8 | **~18.6 GB → KHÔNG vừa ⇒ cần stream/Q3** |

**Bốn trụ kỹ thuật:**
1. **Expert offloading** — expert không dùng nằm ở RAM/đĩa; chỉ expert được router chọn lên VRAM.
2. **Expert cache (LRU/LFU)** — giữ expert "nóng" trong VRAM. Cache hit ⇒ 0 I/O.
3. **Routing prediction / prefetch** — đọc trước expert mà *lớp kế* sẽ chọn, che latency.
   MoE có tính **định xứ theo lớp**: expert lớp n+1 phụ thuộc hidden state lớp n ⇒ dự đoán được.
4. **Expert reuse** — tinh chỉnh router để tăng trùng lặp expert giữa các token (ReMoE).

**Các thuật toán trong tài liệu (mô tả, chưa tự kiểm chứng):** FloE (offload theo tần suất),
EdgeMoE (offload expert ra đĩa cho edge), ReMoE (router tăng reuse), MoE-APEX (prefetch),
MergeMoE (gộp expert ít dùng), BigLittle-MoE (lượng tử hóa expert theo mức dùng).

**Trần tốc độ trên máy này (suy ra):** decode MoE ~ memory-bandwidth-bound.
- DeepSeek-V2-Lite, expert **resident trong RAM** (40 GB/s): 2.4B × 0.5 B ≈ **1.2 GB/token**
  → trần ≈ **~33 tok/s**; nếu expert nằm VRAM (192 GB/s) thì trần cao hơn nhiều.
- Nếu expert trong RAM nhưng **cache miss ⇒ đọc từ NVMe** (2 GB/s) → **~1.7 tok/s**. Chênh **~20×**.
  ⇒ **Mục tiêu số 1 của dự án là tối đa cache hit**, không phải "stream nhanh".

> **Kết luận thiết kế:** chọn **DeepSeek-V2-Lite** làm model Phase-0/1 (vừa RAM, chạy được thật);
> **Qwen3-30B-A3B** là mục tiêu Phase-2 dưới dạng "chứng minh vượt RAM bằng streaming + expert cache".

---

## 1.c CPU–GPU–NPU Co-execution

**Cơ chế.** GPU không chờ CPU: **GPU tính layer N trong khi CPU đọc/tính layer N+1**. Chia tải theo
bandwidth thời gian thực (q* policy của FreeToken): PCIe chậm ⇒ đẩy nhiều layer sang CPU; PCIe nhanh
⇒ dồn về GPU.

**Layer-wise scheduling:**
```
[CPU: prefetch layer N+1]  ║  [GPU: compute layer N]   →  đổi buffer
```

**Trên máy này (đo được):** có dGPU (NVIDIA) + iGPU (Intel Iris Xe), nhưng **không có API hợp nhất**
để hai GPU chia một model (khác HeteroMosaic trên SoC di động nơi CPU/iGPU/NPU chung bộ nhớ).
⇒ Thiết kế thực tế: **CPU + dGPU co-execution**; iGPU/NPU để dành cho lộ trình mobile (NNAPI/Core ML).

---

## 1.d Quantization cho thiết bị di động

| Kỹ thuật | Bit | Ứng dụng | Ghi chú |
|---|---|---|---|
| Block-wise quant (AirLLM) | 4-bit | giảm lượng đọc đĩa | block độc lập ⇒ mmap từng block |
| AWQ | 4-bit | mobile | activation-aware, giữ chất lượng |
| GPTQ | 4-bit | mobile | post-training, nhanh |
| QAT (Squat) | sub-8-bit | SLM | train kèm lượng tử hóa |
| Token-adaptive | 3–4-bit | mobile | đổi bit theo token |

**Trên máy này:** dùng **GGUF K-quants + imatrix** (hệ sinh thái llama.cpp có sẵn). Cân nhắc:
- DeepSeek-V2-Lite: `Q4_K_M` (~10 GB, vừa RAM) là mặc định; `Q3_K_M` (~8 GB) nếu cần chừa RAM cho KV.
- Qwen3-30B-A3B: `Q3_K_M` (~14.8 GB) là mức thấp nhất còn chấp nhận được, vẫn **quá** 11 GiB ⇒
  buộc phải có offloading.

---

## 1.e KV Cache Management

**Cơ chế.** KV cache lớn tuyến tính theo context → ăn hết RAM/VRAM trên máy yếu.
- **PagedAttention (vLLM):** chia KV thành page, cấp phát theo nhu cầu, giảm phân mảnh.
- **Compression / quantization:** KV xuống 8-bit hoặc 4-bit (gần như miễn phí chất lượng).
- **Eviction:** bỏ KV ít quan trọng (H2O, sliding window).
- **Semantic anchor (FreeToken):** lưu checkpoint trạng thái hồi quy để không phải tính lại khi context đổi.

**Trên máy này:** mặc định `--cache-type-k q8_0 --cache-type-v q8_0` trong llama.cpp; bật eviction
khi context > 4k trên model 10 GB.

---

## 1.f Speculative Decoding trên thiết bị yếu

**Cơ chế.** Draft model nhỏ sinh `γ` token, target model verify một lượt.

```
tăng tốc ≈ (1 − α) / (1 − α·β)      α = tỉ lệ draft đúng, β = chi phí draft/target
```

**SS-MoE:** dùng chính model nhỏ hơn (hoặc expert nhỏ) làm draft. **Draft-then-verify:** model nhỏ
local sinh nháp, model lớn verify.

**Trên máy này:** draft = một dense **Qwen2.5-0.5B Q4 (~0.4 GB)**; target = DeepSeek-V2-Lite.
Hai model phải nằm chung ngân sách RAM 11 GiB ⇒ khả thi, nhưng chỉ đáng làm ở **Phase 4**.

---

## 1.g Nén model khác

- **Pruning** (structured/unstructured): bỏ head/neuron/expert ít dùng.
- **Low-rank (SVD)**: tách ma trận W ≈ U·Σ·V, giảm tham số.
- **Distillation**: model lớn → SLM chạy mobile.
- **Weight sharing / n-gram embedding compression**: nén embedding table (áp dụng cho vocab lớn).

Trên máy này ưu tiên **pruning expert ít dùng** (song song với MergeMoE) hơn là SVD, vì MoE thưa sẵn
nên cắt expert cho lợi ích bộ nhớ trực tiếp.

---

## 2. Bảng tổng hợp: kỹ thuật nào cho lợi ích gì trên máy mục tiêu

| Kỹ thuật | Giảm | Tăng | Chi phí | Ưu tiên 1 tháng |
|---|---|---|---|---|
| MoE expert offloading + cache | RAM/VRAM lớn | cache hit | phức tạp | ⭐⭐⭐ (lõi) |
| Routing prediction/prefetch | latency I/O | throughput | vừa | ⭐⭐⭐ |
| Quantization Q4/Q3 + KV q8 | bộ nhớ ~4× | — | thấp | ⭐⭐⭐ |
| Layer-wise streaming | RAM (vượt RAM) | — | I/O trần đĩa | ⭐⭐ (chỉ khi > RAM) |
| CPU–GPU co-execution | — | throughput 10–80%* | cao | ⭐⭐ |
| Speculative decoding | — | 1.5–2.5×* | vừa | ⭐ |
| Pruning/SVD/distill | bộ nhớ | chất lượng↓ | rất cao | ✗ (ngoài 1 tháng) |

\* khoảng giá trị **theo công bố của FreeToken/ tài liệu**, chưa đo trên máy này.
