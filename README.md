# vipera

**Lớp tối ưu file model AI (local LLM) để chạy trên máy yếu** — bắt đầu từ **máy tính cá nhân**,
mở rộng ra **điện thoại**.

`vipera` không phải một runtime mới. Nó là **một lớp nằm giữa file model và runtime**: nhận một
artifact (GGUF/safetensors), **viết lại thành artifact đã tối ưu** + một *manifest* cho runtime biết
cái gì nằm ở đâu (VRAM/RAM/đĩa), thứ tự nạp, và expert nào "nóng".

Lấy cảm hứng từ [FreeToken](https://github.com/FlashML-org/FreeToken) (MoE serving trên consumer
hardware) và [AirLLM](https://github.com/lyogavin/airllm) (layer-wise streaming) — nhưng hai dự án
đó giả định nhiều GPU/RAM. `vipera` nhắm đúng vùng khó: **model vừa RAM nhưng quá VRAM, đĩa chậm**.

> Trạng thái: **đang thiết kế (Phase 0)**. Chưa có code chạy. Số in đậm là đo trên máy mục tiêu;
> số của FreeToken/AirLLM là **theo công bố của họ**, chưa kiểm chứng lại.

---

## Ý tưởng: tối ưu ở **tầng file**, không chỉ ở runtime

Tối ưu runtime (cache expert, prefetch) chỉ giúp được tới trần I/O. `vipera` đi thêm một bước:
**sắp đặt lại chính file model** để runtime yếu đọc ít hơn và đọc tuần tự hơn.

| Transform (tầng file) | Đọc file gốc | Xuất ra | Lợi ích trên máy yếu |
|---|---|---|---|
| **Expert hot/cold reorder** | trace routing | epoch layout lại | expert nóng nằm liền kề ⇒ prefetch 1 lần đọc |
| **Residency manifest** | model + hardware | `*.vipera.json` | runtime biết chia VRAM/RAM/đĩa không cần đoán |
| **Block-wise quant** | safetensors | GGUF block 4-bit | stream từng block qua `mmap` |
| **Expert pruning/merge** | tần suất dùng | ít expert hơn | nhỏ hơn, vừa RAM hơn |
| **KV placement hints** | cấu hình | manifest | KV ở đâu, quant bao nhiêu bit |
| **Chunk alignment** | model | căn theo page/đĩa | tránh đọc thừa trên NVMe DRAM-less |

## Máy mục tiêu (đo 12/10)

| | |
|---|---|
| CPU | Intel i5-11300H — 4C/8T, ≤ 4.4 GHz |
| RAM | 15 GiB, khả dụng ~10 GiB, **không swap** |
| GPU | NVIDIA **GTX 1650 Max-Q, 4 GB VRAM** (compute 7.5) |
| iGPU | Intel Iris Xe (TigerLake) |
| Disk | WD SN530 NVMe **DRAM-less**, 404 GB trống |
| OS | Linux |

## Model mục tiêu

| Phase | Model | Tổng / active | Q4_K_M (ước lượng) | Vừa RAM 11 GiB? |
|---|---|---|---|---|
| P0–P2 | **DeepSeek-V2-Lite** | 15.7B / 2.4B | ~9.5–10.4 GB | ✅ |
| stretch | Qwen3-30B-A3B | 30.5B / 3.3B | ~18.6 GB | ❌ (cần streaming) |

**Con số quyết định:** với DeepSeek-V2-Lite, expert nằm RAM → trần ≈ **~33 tok/s**; nhưng cache
miss đọc từ NVMe (DRAM-less ~2 GB/s) → **~1.7 tok/s**. Chênh **~20×** ⇒ mục tiêu số 1 của vipera là
**giảm số lần đọc đĩa** cho expert.

## Tài liệu

- [`docs/OPTIMIZER.md`](docs/OPTIMIZER.md) — **lớp tối ưu file model** (định vị chính của vipera).
- [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — phân tích thuật toán (streaming, MoE offloading,
  co-execution, quantization, KV cache, speculative decoding, nén model).
- [`docs/DESIGN.md`](docs/DESIGN.md) — kiến trúc, so sánh FreeToken/AirLLM, roadmap 1 tháng, tech
  stack, KPI, rủi ro, **kế hoạch 7 ngày**.

## Phạm vi (1 tháng, 0đ)

- ✅ PC-first (không có máy Android để test ⇒ "phone" chỉ ở mức thiết kế/portability).
- ✅ GGUF + llama.cpp, Python cho tầng optimizer.
- ❌ Không: ONNX/ExecuTorch/LiteRT runtime, gRPC, distillation trong tháng này.
