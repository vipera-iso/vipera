# L2/L3 — Các lớp tối ưu có thật sự làm gì? (số đo, không phải suy đoán)

**Ngày:** 2026-10-21 · Máy: 4C/8T, 15 GiB RAM, CPU-only · Model: `qwen2.5-0.5b-instruct-q4_k_m`
(24 layer, 14 head / 2 KV head, head_dim 64) · Prompt cố định ~470 token · Engine tạo mới
cho mỗi dòng nên `load s` là thời gian nạp thật.

Chạy bằng `tools/bench_layers.py`; dữ liệu thô trong `kv_backends.json`, `fa_cost.json`, `kv_16k.json`.

```bash
.venv/bin/python tools/bench_layers.py --model qwen2.5-0.5b-instruct-q4_k_m \
  --ctx 4096 --max-tokens 64 --backend inprocess server --kv f16 q8_0 q5_1 iq4_nl
```

## A. L2 — KV cache dtype × backend (ctx 4096, sinh 64 token)

| backend | kv | FA planner đặt | load s | TTFT ms | gen t/s | token sinh |
|---|---|---:|---:|---:|---:|---:|
| inprocess | f16 | off | 0.67 | 464 | **67.21** | 64 |
| inprocess | q8_0 | **on** | 0.22 | 597 | 64.97 | 64 |
| inprocess | q5_1 | **on** | 0.22 | 549 | 57.49 | 64 |
| inprocess | iq4_nl | **on** | 0.20 | 549 | 55.05 | 47 |
| server | f16 | off | 0.54 | 579 | 66.28 | 64 |
| server | q8_0 | **on** | 0.50 | 533 | 66.24 | 64 |
| server | q5_1 | **on** | 0.50 | 650 | 61.26 | 6 |
| server | iq4_nl | **on** | 0.50 | 687 | 55.93 | 64 |

Đọc:

- **Planner bật flash-attention đúng như thiết kế**: f16 → off, mọi dtype lượng tử → on.
  Không có dòng nào `eff=on` khi `kv=f16` với `--fa auto`.
- **q8_0 gần như miễn phí:** −3.3% gen t/s (67.2 → 65.0) để đổi **một nửa KV RAM**.
  Đây là mặc định đúng cho máy ít RAM.
- **q5_1 / iq4_nl đắt hơn:** −14% / −18% gen t/s. Ở 0.5B, KV chỉ ~50 MB nên tiết
  kiệm RAM không đáng giá trị tốc độ mất đi; chỉ nên dùng khi KV là thứ chặn ctx.
- **`token sinh` khác nhau là tín hiệu chất lượng, không phải lỗi:** iq4_nl dừng ở 47
  token, `server q5_1` dừng ở 6 token — KV 4–5 bit đổi phân bố xác suất đủ để model
  kết thúc sớm. q8_0 sinh đủ 64 như f16 ở cả hai backend.

## B. L2 — flash-attention trên CPU có thật sự chậm? (kv f16)

Giả định cũ trong `planner.py`/`OPTIMIZE-hermes4.md`: "FA + KV lượng tử trên CPU =
chậm kinh khủng" (đo trên Hermes-4 @65536: 0.1–0.2 tok/s). Đo lại bằng cách ép
`--fa on` với f16, hai độ dài sequence:

**Prompt ~470 token, ctx 4096:**

| backend | FA | TTFT ms | gen t/s |
|---|---|---:|---:|
| inprocess | auto (off) | 464 | 66.71 |
| inprocess | **on** | 425 | 66.09 |
| inprocess | off | 464 | 66.62 |
| server | auto (off) | 543 | 66.12 |
| server | **on** | 566 | 66.91 |
| server | off | 579 | 65.48 |

**Prompt ~3,760 token (repeat 8), ctx 8192:**

| FA | TTFT ms | gen t/s |
|---|---:|---:|
| auto (off) | 3549 | 58.58 |
| **on** | **3197** | 56.78 |
| off | 3260 | 60.31 |

Đọc: ở 0.5B và sequence ≤ ~4k, FA trên CPU **không đắt** — TTFT còn thấp hơn (3.20 s
vs 3.26 s), decode chậm 6%. Con số "50× chậm" của Hermes-4 là ở **ctx 65,536 với KV
lượng tử trên 14B**, một chế độ khác hẳn; phép đo này **không** bác bỏ nó.

Vì vậy: planner giữ nguyên hành vi bảo thủ (f16/bf16/f32 không bật FA) là hợp lý —
không có lợi ích đo được, và ở ctx lớn nó là đường đã biết rất chậm. Nhưng ghi chú
"FA luôn chậm" trong tài liệu là **quá rộng**: nó chỉ đúng ở ctx lớn / model lớn.

## C. L2 — KV RAM thật sự tiết kiệm bao nhiêu (tính bằng `planner._estimate_kv_cache_gb`)

Hermes-4-14B (40 layer, 8 KV head, head_dim 128):

| ctx | f16 | bf16 | q8_0 | q5_1 | q4_0 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.62 GB | 0.62 GB | 0.31 GB | 0.23 GB | 0.16 GB |
| 8,192 | 1.25 GB | 1.25 GB | 0.62 GB | 0.47 GB | 0.31 GB |
| 32,768 | 5.00 GB | 5.00 GB | 2.50 GB | 1.88 GB | 1.25 GB |

Đo thực tế ở ctx 16384 trên qwen0.5b không phân biệt được (KV chỉ ~100 MB f16 vs
~53 MB q8_0, nằm dưới nhiễu RSS), nên bảng trên là **số tính từ công thức đã có
test** (`tests/test_planner.py::test_kv_estimate_*`), không phải số đo RSS.

## D. L3 — backend `server` so với `inprocess`

| | inprocess | server |
|---|---|---|
| load (0.5B) | 0.20–0.67 s | 0.50 s (spawn + chờ `/health`) |
| gen t/s | 65–67 | 66–67 |
| streaming | generator trong process | SSE từ `llama-server` |
| cách ly | chung GIL/RAM với gateway | process riêng, chết không kéo gateway |

**Backend `server` không mất throughput** (66.2 vs 67.2 t/s, ~1.5% trong nhiễu) và
trả giá đúng 0.3 s khởi động trên model 0.5B. Lớp này hoạt động đúng như thiết kế:
đổi backend không đổi chất lượng sinh, chỉ đổi chỗ chạy.

## Tóm lại

| Lớp | Có tối ưu thật không? | Bằng chứng |
|---|---|---|
| L2 KV dtype | **Có** — q8_0 nửa RAM, −3% tốc độ; q5_1/iq4_nl đắt hơn, chất lượng đổi | bảng A |
| L2 FA | **Rẻ ở sequence ≤4k** (+0%); chưa do với Hermes 64k nên không kết luận chung | bảng B |
| L2 KV estimate | **Có** — planner tính đúng theo dtype, không dùng heuristic 1.5 GB | bảng C + test |
| L3 backend | **Có** — cùng tốc độ, +0.3 s khởi động, cách ly process | bảng D |
| L1a imatrix | **Có ở Q3** (−5.18%), không kết luận được ở Q4 (−0.96%) | `../l1a2_results/REPORT.md` |

## Hạn chế

- 0.5B, prompt ~470 token, 1 lần/dòng. Chênh lệch <5% là nhiễu; đừng đọc 67.21 vs
  66.28 thành "inprocess nhanh hơn".
- CPU-only. Trên GPU, FA là đường nhanh **hơn** nhiều — kết luận B không áp dụng cho GPU.
- KV RAM ở bảng C là số tính, không phải RSS đo được.
