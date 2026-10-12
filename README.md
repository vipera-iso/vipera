# edge-llm

Chạy **LLM/MoE trên thiết bị yếu** — lấy cảm hứng từ [FreeToken](https://github.com/FlashML-org/FreeToken)
(MoE serving trên consumer hardware) và [AirLLM](https://github.com/lyogavin/airllm) (layer-wise streaming).

Khác biệt định vị: hai dự án kia giả định **nhiều GPU/RAM**. `edge-llm` nhắm đúng vùng khó —
**model vừa RAM nhưng quá VRAM, đĩa chậm** — và tập trung vào **bộ đệm expert + dự đoán routing**
để giảm I/O, thay vì chỉ "stream nhanh".

> Trạng thái: **đang thiết kế (Phase 0)**. Chưa có code chạy. Số trong tài liệu là **đo trên máy
> mục tiêu**; con số của FreeToken/AirLLM là **theo công bố của họ**, chưa kiểm chứng lại.

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

## Tài liệu

- [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — phân tích thuật toán (layer-wise streaming, MoE expert
  offloading, CPU–GPU co-exec, quantization, KV cache, speculative decoding, compression).
- [`docs/DESIGN.md`](docs/DESIGN.md) — kiến trúc, so sánh FreeToken/AirLLM, roadmap 1 tháng, tech
  stack, KPI, rủi ro, **kế hoạch 7 ngày**.

## Giới hạn phạm vi (1 tháng, 0đ)

- ✅ PC-first (không có máy Android để test ⇒ "mobile" chỉ ở mức thiết kế).
- ✅ GGUF + llama.cpp, Python cho tầng cache/scheduler.
- ❌ Không: ONNX/ExecuTorch/LiteRT runtime, gRPC, pruning/SVD/distillation trong tháng này.

## Bắt đầu (dự kiến)

```bash
# Phase 0 baseline (chưa có script — sẽ thêm ở Ngày 1)
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp && cmake --build llama.cpp/build -j
# bench sẽ nằm ở ./bench
```
