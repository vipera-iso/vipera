# DESIGN — edge-llm: chạy LLM/MoE trên thiết bị yếu

> Kế hoạch thiết kế hệ thống. Phần thuật toán sâu ở [`ALGORITHMS.md`](ALGORITHMS.md).
> Trả lời bằng tiếng Việt; số in đậm là đo trên máy hoặc suy ra từ đo.

## Bối cảnh & giả định đã chốt

| Hạng mục | Quyết định | Nguồn |
|---|---|---|
| Thiết bị mục tiêu | **PC cá nhân** (i5-11300H, 15 GiB RAM, GTX 1650 4 GB, NVMe) | user chọn + đo |
| Model mục tiêu | **MoE nhỏ**: DeepSeek-V2-Lite (Phase 0/1), Qwen3-30B-A3B (stretch) | user chọn |
| Ngân sách | **0đ** — chỉ open-source, không thuê cloud | user chọn |
| Thời gian | **1 tháng** | user chọn |
| Giấy phép | Open-source, offline-first, cross-platform | prompt gốc |
| Mobile | **Giai đoạn sau** (không có trong 1 tháng này) | suy ra từ "PC cá nhân" |

**Giả định ghi rõ:** không có thiết bị Android để test trong tháng này ⇒ mọi "mobile" chỉ ở mức
thiết kế/portability, **không** phải kết quả đo. Nếu giả định sai, báo để đổi scope.

---

## 2. Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────┐
│                         API GATE                                  │
│  OpenAI-compatible /v1/chat/completions   ·   gRPC (sau)          │
└───────────────▲──────────────────────────────────────────────────┘
                │ request
┌───────────────┴──────────────────────────────────────────────────┐
│                      RUNTIME ROUTER                               │
│  chọn runtime theo (model, hardware):                             │
│    dense+GGUF → llama.cpp   ·  mobile → ExecuTorch / LiteRT       │
│    ONNX → ORT               ·  fallback → Python reference        │
└───────┬───────────────────────────────┬──────────────────────────┘
        │                               │
┌───────▼─────────┐           ┌─────────▼──────────────────────────┐
│ MODEL MANAGER   │           │      MEMORY MANAGER  ★ lõi ★        │
│ download/convert│           │  VRAM budget (4 GB)                 │
│ cache/quantize  │           │  ├─ attention + shared  (resident)  │
│ (GGUF, imatrix) │           │  ├─ expert cache (LRU/LFU) ★        │
└───────┬─────────┘           │  ├─ KV cache (paged, q8)            │
        │                     │  └─ host/disk spill (mmap, NVMe)    │
        │                     └─────────┬──────────────────────────┘
        │                               │
┌───────▼───────────────────────────────▼──────────────────────────┐
│                    EXECUTION ENGINE                               │
│  Expert scheduler: routing predictor → prefetch → compute         │
│  CPU + dGPU co-execution (layer-wise overlap)                     │
│  Backend: llama.cpp (MoE CPU-offload) → custom expert-cache layer │
└──────────────────────────────────────────────────────────────────┘
```

**Thành phần & trạng thái trong 1 tháng:**

| Thành phần | Vai trò | Trong 1 tháng? |
|---|---|---|
| Memory Manager (expert cache + KV) | **đóng góp lõi**, khác biệt so với có sẵn | ✅ Phase 2 |
| Execution Engine (expert scheduler + prefetch) | dự đoán expert, overlap I/O | ✅ Phase 2 |
| Model Manager | tải/convert GGUF, cache | ✅ Phase 0 |
| Runtime Router | chọn backend | ⚠️ chỉ 2 nhánh (llama.cpp + Python ref) |
| API Gate | OpenAI-compatible | ✅ Phase 5 (bản tối giản) |
| Mobile runtime | NNAPI/Core ML | ❌ roadmap |

---

## 3. So sánh với FreeToken và AirLLM

| Tiêu chí | FreeToken | AirLLM | **edge-llm (đề xuất)** |
|---|---|---|---|
| Trọng tâm | MoE serving trên PC consumer | layer-wise streaming | **MoE offload + expert cache trên máy yếu** |
| Đơn vị offload | expert (fractional) | layer | expert + block |
| Điểm mạnh | CPU–GPU co-exec, token-level split | chạy model > VRAM đơn giản | hợp nhất: streaming khi > RAM, cache khi vừa RAM |
| Hạn chế | hướng GPU mạnh; chưa tối ưu máy 4 GB | I/O-bound, tốc độ thấp | (mới) |
| Tăng tốc công bố | CPU–GPU 84–96% (T4), 11–89% (A10) | chạy 671B trên 12 GB | **mục tiêu: ≥ 2× token/s vs llama.cpp CPU-offload mặc định trên máy này** |
| Phù hợp mobile | một phần | không | một phần (thiết kế tách backend) |

**edge-llm cải tiến gì:** FreeToken tối ưu cho GPU consumer *đủ mạnh*; AirLLM tối ưu *vượt VRAM*.
Cả hai giả định nhiều RAM. Máy mục tiêu ở đây (**4 GB VRAM + 15 GB RAM, không swap, NVMe DRAM-less**)
rơi đúng vào vùng mà **cả hai đều dở**: model vừa RAM nhưng quá VRAM, và đĩa chậm. Đóng góp của
edge-llm là **bộ đệm expert thông minh + dự đoán routing để giảm I/O**, chứ không phải "stream nhanh".

---

## 4. Roadmap (co lại cho 1 tháng / 0đ)

| Phase | Nội dung | Kết quả đo được | Tuần |
|---|---|---|---|
| **P0** | Baseline: build llama.cpp, chạy DeepSeek-V2-Lite Q4 (CPU-only & partial GPU), đo tokens/s, RAM/VRAM | bảng baseline trung thực | 1 |
| **P1** | Quant + KV: so Q4_K_M/Q3_K_M, KV q8; đo chất lượng (perplexity) vs tốc độ | bảng đánh đổi | 1–2 |
| **P2** | ★ **Expert cache (LRU/LFU) + routing prediction/prefetch** trên MoE | cache hit %, tokens/s vs P0 | 2–3 |
| **P3** | CPU + dGPU co-execution (overlap layer N/N+1) | tokens/s vs P2 | 3 |
| **P4** | KV paged + speculative (Qwen2.5-0.5B draft) | speedup | 4 (nếu kịp) |
| **P5** | API OpenAI-compatible + CLI + model manager | `curl` chạy được | 4 |

**Cắt khỏi 1 tháng (ghi vào roadmap v1.1+):** mobile runtime (NNAPI/Core ML/ExecuTorch),
ONNX Runtime, gRPC, pruning/SVD/distillation, Qwen3-30B-A3B đầy đủ.

**Tiêu chí "thành công tối thiểu" cuối tháng:** chạy DeepSeek-V2-Lite trên máy này, có **API
OpenAI-compatible**, và **chứng minh bằng số** rằng expert cache + prefetch thắng baseline rõ ràng
(lý tưởng: hit ≥ 60%, token/s tăng ≥ 1.5×) — nếu không thắng, báo cáo trung thực là không thắng.

---

## 5. Tech stack đề xuất (theo ngân sách 0đ)

| Tầng | Chọn | Lý do |
|---|---|---|
| Core engine | **C++ (llama.cpp fork/patch)** | tái dùng kernel + GGUF + CUDA sẵn có |
| Điều phối/cache | **Python 3** | nhanh thử nghiệm, có `mmap`, `ctypes` |
| Kernel tùy biến | C++ + CUDA (chỉ khi cần) | `nvcc` chưa cài — cài sau nếu cần |
| Build | `cmake` + `gcc` (đã có) | không cần cargo |
| Model format | **GGUF** (+ imatrix) | hệ sinh thái sẵn, hỗ trợ MoE offload |
| Mobile (sau) | Kotlin (NNAPI), Swift (Core ML), ExecuTorch | roadmap, không trong 1 tháng |
| Runtime khác | ONNX/MLC/ORT | để dành router, chưa làm |

---

## 6. KPI (đo baseline riêng, không mượn số)

| Chỉ số | Cách đo |
|---|---|
| Tokens/s (prefill + decode) | đếm token thật / wall-clock, ≥ 3 lần chạy |
| RAM đỉnh (RSS) | `/usr/bin/time -v` hoặc `ps` sampling |
| VRAM đỉnh | `nvidia-smi --query-gpu=memory.used` |
| Cache hit % (expert) | log bộ đệm, hit/tổng truy cập expert |
| I/O mỗi token | `/proc/<pid>/io` read_bytes / token |
| Latency first token | timestamp trước/sau token đầu |
| Perplexity | `llama-perplexity` trên wiki test |
| Nhiệt/điện | `nvidia-smi` + `powertop` (ưu tiên thấp trên PC) |

> **Không** dùng "Recall@5" hay số của FreeToken/AirLLM làm KPI của dự án này — đó là số của họ.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| **I/O bottleneck** (NVMe DRAM-less ~2 GB/s) | giữ expert nóng trong VRAM/RAM; đo hit trước khi tối ưu đọc |
| **Model quá RAM** (Qwen3-30B-A3B 18.6 GB > 11 GiB) | chọn DeepSeek-V2-Lite cho P0–P2; Qwen3 chỉ khi có streaming |
| **Không swap** ⇒ OOM chết tiến trình | đặt `--mlock` cẩn thận, giới hạn RSS, `systemd-run` giới hạn |
| **Cài llama.cpp CUDA** cần `nvcc`? | build CPU trước (đã có gcc/cmake); CUDA chỉ khi cần |
| Đĩa 404 GB trống đủ cho vài model ~30 GB | theo dõi dung lượng |
| Chất lượng giảm khi Q3/lượng tử hóa mạnh | đo perplexity, không hạ bit nếu tăng > 0.3 |
| Bản quyền model | chỉ dùng model license permissive (DeepSeek/Qwen đều cho phép) |
| Nhiệt/throttle laptop | đo nhiệt, tránh chạy liên tục > 30 phút |
| "Thắng" không đo được ⇒ tự lừa | chốt tiêu chí baseline trước, báo cáo cả khi thua |
| NPU/Android không có máy test | ghi rõ là "chưa kiểm chứng", không nhận là đã làm |

---

## 8. Next 7 days — action plan

**Ngày 1 — nền tảng đo**
- [ ] Clone & build `llama.cpp` (CPU-only trước) bằng `cmake`.
- [ ] Viết `bench/` (script Python): đo token/s, RSS, VRAM, I/O per token, chuẩn hóa output.
- [ ] Chốt định dạng báo cáo baseline (bảng + JSON).

**Ngày 2 — model P0**
- [ ] Tải DeepSeek-V2-Lite GGUF `Q4_K_M` (~10 GB) + lưu checksum.
- [ ] Chạy baseline: CPU-only (`-ngl 0`), partial (`-ngl 99 --n-cpu-moe`), ghi số.
- [ ] Xác nhận VRAM/RAM thực tế, ghi bảng.

**Ngày 3 — quant & KV (P1)**
- [ ] So `Q4_K_M` vs `Q3_K_M`; KV `f16` vs `q8_0` vs `q4_0`.
- [ ] Đo perplexity từng cấu hình (wiki-test ngắn).
- [ ] Bảng đánh đổi chất lượng/tốc độ/bộ nhớ.

**Ngày 4 — thiết kế expert cache (P2)**
- [ ] Viết spec: khóa cache, chính sách LRU vs LFU vs hybrid, kích thước theo VRAM còn trống.
- [ ] Instrument llama.cpp/offload để **log expert nào được chọn** (dữ liệu cho dự đoán).
- [ ] Thu trace routing ≥ 1k token làm dataset.

**Ngày 5 — dự đoán routing (P2)**
- [ ] Phân tích trace: entropy routing, trùng lặp expert giữa lớp liền kề, tỉ lệ hit tiềm năng.
- [ ] Prototype predictor đơn giản (freq + last-used) đo hit%.

**Ngày 6 — tích hợp & đo (P2)**
- [ ] Nối cache + prefetch vào engine; chạy lại cùng benchmark ngày 2.
- [ ] So P0 vs P2: hit %, token/s, I/O/token. Ghi thẳng cả khi thua.

**Ngày 7 — chốt tuần & quyết định**
- [ ] Viết `docs/RESULTS-W1.md` với số thật.
- [ ] Quyết định có đủ cơ sở làm P3 (co-exec) hay quay lại tối ưu cache.
- [ ] Commit + tag `w1`.

---

## Sơ đồ luồng một lượt decode (mục tiêu P2)

```
token t
  │
  ├─ router layer n  ──► chọn expert {e1,e2,...}
  │        │
  │        ├─ cache hit? ──► compute ngay (VRAM)
  │        └─ miss ──► nạp từ RAM/đĩa (async) ──┐
  │                                              │
  ├─ predictor: dự đoán expert layer n+1 ──► prefetch (che latency)
  │
  └─ KV cache: cấp page cho token mới (không copy cả KV)
```
