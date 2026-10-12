# OPTIMIZER — lớp tối ưu file model

> Đây là **định vị chính** của vipera: không viết runtime mới, mà **viết lại file model** để runtime
> yếu đọc ít hơn và đọc tuần tự hơn. Runtime (llama.cpp) là *người tiêu thụ* artifact do vipera xuất ra.

---

## 1. Vì sao tối ưu ở tầng file

Trên máy mục tiêu (4 GB VRAM, 15 GiB RAM, NVMe DRAM-less), chi phí decode MoE bị chi phối bởi
**số byte phải đọc cho mỗi token**. Tối ưu runtime (cache, prefetch) chỉ đẩy tới trần I/O; nó
**không đổi được** việc file model xếp expert nóng rải rác khắp đĩa.

Chia nhỏ vấn đề:

```
chi phí token  =  (bytes đọc từ VRAM/RAM)  +  (bytes đọc từ đĩa)
                  └─ nhanh, không đổi ─┘     └─ CHẬM (2 GB/s) — đây là chỗ tối ưu ─┘
```

**Mục tiêu của lớp tối ưu:** giảm *bytes đọc từ đĩa*, và khi buộc phải đọc thì đọc **tuần tự**.

---

## 2. Pipeline

```
 safetensors / GGUF gốc
        │
        ▼
 ┌─────────────────┐   (1) phân tích
 │  ANALYZER       │   • trace routing (expert nào được chọn theo layer/token)
 │                 │   • độ phổ biến expert  • nhóm lỗi thời • kích thước tensor
 └────────┬────────┘
          ▼
 ┌─────────────────┐   (2) biến đổi (ghi rõ dữ liệu vào/ra)
 │  TRANSFORMER    │   • reorder expert  • prune/merge  • block-quant  • căn chunk
 └────────┬────────┘
          ▼
 ┌─────────────────┐   (3) đóng gói
 │  PACKER         │   • artifact .gguf đã tối ưu
 │                 │   • `*.vipera.json` = residency manifest
 └────────┬────────┘
          ▼
   runtime (llama.cpp) đọc manifest → biết cái gì ở đâu
```

---

## 3. Residency manifest — `*.vipera.json`

Hợp đồng giữa optimizer và runtime. Nội dung dự kiến:

```jsonc
{
  "vipera_version": 1,
  "source": { "model": "deepseek-v2-lite", "quant": "Q4_K_M", "sha256": "..." },
  "target": { "vram_mb": 4096, "ram_mb": 11264, "disk": "nvme-dramless" },
  "placement": {
    "resident_vram": ["attn.*", "shared_experts.*", "norm.*"],
    "resident_ram":  ["hot_experts/*"],        // ví dụ 60% expert nóng
    "spill_disk":    ["cold_experts/*"]        // đọc theo nhu cầu
  },
  "expert_order": "by_frequency",              // thứ tự trong file
  "routing": { "trace_tokens": 4096, "topk": 6, "hit_potential": 0.71 },
  "kv_cache": { "type_k": "q8_0", "type_v": "q8_0", "pages": 256 }
}
```

Runtime đọc manifest ⇒ **không phải dò/đoán** khi khởi động; tránh tự động chọn `-ngl`/`--n-cpu-moe`
một cách mù quáng.

---

## 4. Các transform, chi tiết

### T1. Expert hot/cold reorder
- **Vào:** trace routing (từ P2) + file gốc.
- **Ra:** file có expert sắp theo tần suất; expert nóng **liền kề trên đĩa**.
- **Vì sao thắng:** NVMe DRAM-less đọc tuần tự tốt hơn nhiều so với random. Nhiều expert nóng liền
  nhau ⇒ **1 lần đọc tuần tự**, không phải N lần đọc rời.
- **Rủi ro:** nếu phân bố expert đổi theo domain prompt, thứ tự tối ưu cho task này lại kém cho task
  khác ⇒ cần reorder theo *nhiều* trace, không tin một trace.

### T2. Residency manifest
- **Vào:** model + cấu hình máy.
- **Ra:** `*.vipera.json` (mục 3).
- **Vì sao thắng:** bỏ bước "đoán" khi chạy; placement tường minh ⇒ ít thử-sai, và nó là **đầu vào**
  cho expert cache ở runtime.

### T3. Block-wise quantization
- **Vào:** safetensors (f16/bf16).
- **Ra:** GGUF block 4-bit (block độc lập, mmap từng block).
- **Vì sao thắng:** giảm bytes cả trên đĩa lẫn RAM; block độc lập ⇒ có thể `mmap` từng mảnh, hợp với
  streaming. (AirLLM dùng đúng ý tưởng này để giảm I/O.)

### T4. Expert pruning / merge
- **Vào:** tần suất dùng expert + đóng góp chất lượng.
- **Ra:** ít expert hơn (bỏ expert gần như chưa bao giờ được chọn; hoặc gộp cặp giống nhau).
- **Vì sao thắng:** MoE thưa sẵn ⇒ cắt expert cho lợi ích bộ nhớ **trực tiếp**.
- **Rủi ro:** **chất lượng tụt âm thầm**. Bắt buộc đo perplexity trước/sau; cắt quá tay ⇒ phải rollback.

### T5. KV placement hints
- **Vào:** cấu hình context mong muốn.
- **Ra:** manifest (kiểu KV, số page).
- **Vì sao thắng:** KV trên máy này dễ ăn hết RAM; đặt tường minh tránh OOM (máy **không có swap**).

### T6. Chunk alignment
- **Vào:** file.
- **Ra:** tensor căn theo page/đĩa.
- **Vì sao thắng:** tránh đọc thừa byte ở rìa page — quan trọng khi đĩa DRAM-less.

---

## 5. Hợp đồng (interfaces)

```python
# vipera/optimizer.py (thiết kế, chưa cài đặt)
class Optimizer:
    def analyze(self, model_path: str, traces: list[str]) -> "Analysis": ...
    def transform(self, model_path: str, plan: "Plan") -> "Artifact": ...
    def write_manifest(self, artifact: "Artifact", target: "TargetHW") -> str: ...
```

CLI dự kiến:

```bash
# phân tích: sinh trace + thống kê expert
vipera analyze models/deepseek-v2-lite-Q4_K_M.gguf --trace bench/prompts.txt -o analysis.json

# tối ưu: ghi artifact + manifest
vipera optimize models/deepseek-v2-lite-Q4_K_M.gguf --target hw/gtx1650.toml -o out/

# xuất ra: out/model.vipera.gguf, out/model.vipera.json
```

`hw/gtx1650.toml` mô tả máy mục tiêu (VRAM/RAM/đĩa) để placement tính được.

---

## 6. Đo lường (bắt buộc trước khi tin bất kỳ transform nào)

| Transform | Cách kiểm |
|---|---|
| T1 reorder | bytes đọc/token trước-sau; hit%; tokens/s |
| T2 manifest | sai số placement (dự đoán vs thực tế VRAM/RAM đỉnh) |
| T3 block-quant | perplexity tăng ≤ 0.3 so với file gốc |
| T4 prune/merge | perplexity trên **tập câu hỏi đa dạng**, không chỉ 1 prompt |
| T5 KV hints | RAM đỉnh, có OOM không |
| T6 align | thời gian nạp lạnh (cold cache) |

**Nguyên tắc:** mỗi transform phải **tự chứng minh** bằng số trước/sau. Transform nào không thắng
trên máy mục tiêu thì **tắt mặc định** và ghi lại là thất bại — không giữ lại cho đẹp.
