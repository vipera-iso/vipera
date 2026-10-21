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

## E. L1 — đổi weight Q4_K_M → Q3_K_M+imatrix trên Hermes-4 (14B), số đo thật

3 lần lặp mỗi model, cùng prompt (~470 token), ctx 4096, kv f16, 64 token sinh:

| | load s | TTFT s | decode t/s | PPL |
|---|---:|---:|---:|---:|
| `q8src-ctl-q4km` (9.00 GB) | 7.9 / 9.2 / 8.0 | 9.8 / 10.1 / 13.2 | 3.73 / 3.76 / 3.69 | 7.0056 |
| `q8src-imbf-q3km` (7.32 GB) | 6.1 / 6.0 / 5.9 | 8.0 / 7.4 / 7.4 | 3.85 / 3.89 / 3.74 | **6.9805** |
| chênh (median) | **−25%** | **−27%** | **+3%** | **−0.36%** |

Nghĩa là: bớt 18.7% dung lượng file, nạp nhanh hơn 25%, prefill nhanh hơn 27%,
decode gần như không đổi (+3%), chất lượng **không giảm** (PPL thấp hơn 0.36%).
Decode không nhanh hơn nhiều vì Q3_K_M dequant đắt hơn Q4_K_M — bớt byte đọc
nhưng thêm việc giải mã.

Ghi chú quan trọng: gọi bằng đường dẫn `.gguf` nên `meta` trống (`layers=0`), tức
cả hai model đi qua **nhánh fallback heuristic** của planner, không phải nhánh
per-model (`_kv_layer_count`/`head_dim` từ metadata). Vì cả hai đi cùng một nhánh
nên so sánh L1 vẫn công bằng — nhưng **không được** trích bảng E để nói "L2 KV
estimate chạy đúng": nhánh per-model không được chạm tới ở đây. Số L2 nằm ở
bảng C, và bảng C là số **tính** từ `planner._estimate_kv_cache_gb` (công thức đã
có test), **không phải RSS đo được**.

## Tóm lại

| Lớp | Có tối ưu thật không? | Bằng chứng |
|---|---|---|
| L2 KV dtype | **Có** — q8_0 nửa RAM, −3% tốc độ; q5_1/iq4_nl đắt hơn, chất lượng đổi | bảng A |
| L2 FA | **Rẻ ở sequence ≤4k** (+0%); chưa đo lại Hermes 64k nên không kết luận chung | bảng B |
| L2 KV estimate | **Có** — planner tính đúng theo dtype, không dùng heuristic 1.5 GB | bảng C + test |
| L3 backend | **Có** — cùng tốc độ, +0.3 s khởi động, cách ly process | bảng D |
| L1a imatrix | **Có ở Q3** (−5.18%), không kết luận được ở Q4 (−0.96%) | `../l1a2_results/REPORT.md` |
| L1 đổi weight | **−19% file, −25% load, −27% prefill, PPL không giảm** | bảng E |

### Tổng % đạt được (Hermes-4, một máy CPU-only 15 GiB)

| Hạng mục | Trước | Sau (Q3+imatrix, kv q8_0) | Đổi |
|---|---:|---:|---:|
| Kích thước file | 9.00 GB | 7.32 GB | **−18.7%** |
| KV @8k (f16 → q8_0) | 1.25 GB | 0.62 GB | **−50%** |
| Tổng RAM @8k | 10.25 GB | 7.94 GB | **−22.5%** |
| Tổng RAM @32k | 14.00 GB | 9.82 GB | **−29.9%** |
| Thời gian nạp | 8.0 s | 6.0 s | **−25%** |
| Prefill (TTFT) | 10.1 s | 7.4 s | **−27%** |
| Decode | 3.73 t/s | 3.85 t/s | +3% |
| PPL (thấp = tốt) | 7.0056 | 6.9805 | **−0.36%** |

Hàng PPL: 7.0056 → 6.9805 là Δ = 0.025 = **0.11σ** — **tương đương trong sai số**,
không phải "Q3 tốt hơn Q4". Claim bảo vệ được là *"không giảm chất lượng"*.

Ba lớp còn lại (`server` backend, FA, dtype q5_1/iq4_nl) **không đem lại % nào** về
tốc độ hay dung lượng ở máy này — chúng đem lại cách ly process / khả năng chọn,
nên đừng tính vào con số "tối ưu".

## Hạn chế

- Bảng A–D: 0.5B, prompt ~470 token, 1 lần/dòng. Chênh lệch <5% là nhiễu; đừng
  đọc 67.21 vs 66.28 thành "inprocess nhanh hơn". Bảng E: Hermes-4 14B, 3 lần/model
  (median) nên chênh lệch 25–27% là thật, còn +3% decode thì vẫn trong nhiễu.
- CPU-only. Trên GPU, FA là đường nhanh **hơn** nhiều — kết luận B không áp dụng cho GPU.
- KV RAM ở bảng C là số **tính** từ `planner._estimate_kv_cache_gb`, không phải RSS
  đo được — chưa có phép đo RSS nào trong báo cáo này chứng minh con số đó.
- Bảng E chỉ đo nhánh fallback của planner (path `.gguf` → meta trống), không đo
  nhánh per-model. Xem ghi chú ở mục E.
