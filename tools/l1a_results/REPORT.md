# L1a — Hiệu ứng imatrix trên Q4_K_M (Hermes-4 + qwen15b) + sanity gate Q3

**Ngày:** 2026-10-21 · **Kết quả:** `STOP — Q3 gate fail (0.2396 < 0.4)` · **Không L1b, không ship pipeline.**

---

## ĐÍNH CHÍNH 2026-10-21 (đọc trước khi dùng bất kỳ số nào bên dưới)

**Nhánh Q4 của L1a v2 không đo được gì.** `llama-quantize` copy nguyên byte khi
type nguồn trùng type đích — `src/llama-quant.cpp`: `bool quantize = cur_type != new_type;`
— và imatrix chỉ được tra cứu *bên trong* nhánh có lượng tử hoá thật. Vì vậy
`Q4_K_M → Q4_K_M` không bao giờ đi qua bộ lượng tử hoá: imatrix không được đọc,
dù file có `--imatrix`. Con số `+0.0000` ở #2/#3/#4 là **no-op cấu trúc**, không
phải "imatrix không có tác dụng ở Q4".

Bằng chứng (script mới `tools/gguf_quant_audit.py`, so từng tensor):

| Cặp file | Tensor đổi type | Tensor đổi data | Kết luận |
|---|---:|---:|---|
| `Hermes Q4_K_M` vs `ctl-hermes-q4km` | 0/443 | **0/443** | copy byte; arm Q4 vô nghĩa |
| `ctl-hermes-q4km` vs `l1a-hermes-q4km` | 0/443 | **0/443** | imatrix không có đường tác động |
| `ctl-qwen15b-q4km` vs `l1a-qwen15b-q4km` | 0/339 | 32/339 | nguồn qwen15b không phải Q4_K_M thuần → có requantize thật |
| `ctl-hermes-q3km` vs `l1a-hermes-q3km` | 0/443 | nhiều | requantize thật, imatrix tác động vào **giá trị** trong cùng type |

Hai diễn giải trong báo cáo gốc là **sai**:

1. *"Q4 = no-op từng byte … imatrix không đổi bất kỳ quyết định nào ở Q4"* —
   không có quyết định nào để đổi, vì không tensor nào được lượng tử lại.
   (Việc type không đổi ở Q3 cũng là bình thường: type do recipe `Q3_K_M`
   chọn, imatrix chỉ tham gia bộ lượng tử hoá giá trị
   — `llama_tensor_quantize_impl(..., imatrix, ...)`.)
2. *"Imatrix thiếu entry `token_embd.weight` → embedding bị loại khỏi
   importance"* — llama.cpp **cố ý** bỏ `token_embd`/`output`:
   `tensor_requires_imatrix()` trả `false` cho hai tensor này, và comment ngay
   trong `llama-quant.cpp` ghi *"tok_embd should be ignored in this case,
   since it always causes this warning"*. Cảnh báo là bình thường.

**Arm Q3 vẫn hợp lệ:** 7.3904 → 7.1508, Δ = **−0.2396 (−3.24%)**, chạy xác định.
Nhưng stderr của PPL trên eval 12,264 token là ±0.2235, nên hiệu ứng này chỉ
cỡ **1σ**: đủ để nói "imatrix có đi vào cuộc", không đủ để kết luận nó tổng
quát hoá sang văn bản khác.

Hệ quả cho kế hoạch: muốn đo hiệu ứng ở Q4 phải requantize từ nguồn có độ phân
giải cao hơn (`Q8_0`/`BF16`), không phải từ chính file Q4_K_M.

---

## Tính toàn vẹn của đo lường (kiểm trước khi đọc số)

- Binary: `llama-perplexity 0.7.0-dev (build 1, commit 797de63)`, sha256 `56c271b8ded0ec4b45e2c329e61ce2b03b46a3a420f9570ffba04a9531f2c398` — **khớp L5a** → so được với baseline 6.9289.
- `grep '!!! PPL RUN FAILED'` = **0** · đủ **8/8 JSON** · mọi run: `ctx=512, ngl=0, chunks=24, n_seq=4, scored_tok=12264`, eval = `wikitext_sample.txt` (sha256 `2a8ef094…`).
- **Determinism:** `orig_run2` = run1 **chính xác** 6.9289 ± 0.22355 (PPL và stderr khớp tới số lẻ cuối) → noise run-to-run = 0. Mọi sai khác PPL giữa 2 file là **thật**, không phải nhiễu máy.
- PPL in ra 4 số lẻ → resolution = 0.0001; các giá trị 0.0000 ở dưới được xác nhận bằng hash tensor (không phải làm tròn).

## 5 con số (ngưỡng đã chốt từ trước)

| # | Số | Giá trị | Ngưỡng | verdict |
|---|---|---|---|---|
| 1 | `orig_run2 − 6.9289` (session noise) | **+0.0000** | < 0.3 | **PASS** — máy ổn, chạy xác định |
| 2 | `control − orig_run2` (requant loss, Hermes Q4) | **+0.0000** | — | quan sát |
| 3 | `l1a − control` (hiệu ứng imatrix Q4, **số chính**) | **+0.0000** | ≥ 0.25 | null — **không đọc để kết luận** (gate #5 fail) |
| 4 | `l1a − orig_run2` (net) | **+0.0000** | — | quan sát — không kết luận |
| 5 | `q3_l1a − q3_control` (Q3 sanity **gate**) | **−0.2396** | \|Δ\| ≥ 0.4 | **FAIL → STOP** |

Hàng phụ (qwen15b, không chạm gate): requant loss `+0.0559`, imatrix effect `−0.0337` (cải thiện nhẹ), net `+0.0222` — đều dưới noise ±0.40.

**Quyết định theo kịch bản đã chốt:** gate #5 fail → `STOP: harness/imatrix cần điều tra`, không kết luận ở Q4, **không đề xuất L1b**, không ship pipeline imatrix.

## Bằng chứng cho điều tra (quan sát, không phải kết luận)

1. **Q4 = no-op từng byte (Hermes):** sha256 data của **toàn bộ 443 tensor** giống nhau tuyệt đối giữa `orig / ctl_q4 / l1a_q4` (`c4c26f3b…`) — requantize Q4_K_M→Q4_K_M trả về weight y hệt gốc, và imatrix **không đổi bất kỳ quyết định nào** ở Q4. File hash khác nhau chỉ do metadata.
2. **Imatrix có vào cuộc ở Q3 (harness không chết):** hash tổng `ctl_q3 ≠ l1a_q3`; `blk.0.attn_q` khác; PPL Q3 cải thiện **đúng chiều** 0.2396 (7.3904 → 7.1508). Cơ chế hoạt động, chỉ **sức mạnh dưới ngưỡng** (0.4).
3. **Imatrix thiếu entry `token_embd.weight`:** log dòng 674 `did not find weights for token_embd.weight` (chỉ lần chạy Q3) → embedding bị loại khỏi importance weighting → một phần làm effect nhỏ.
4. **Caveat đã biết (phải loại trừ trước khi kết luận "harness hỏng"):** imatrix build từ **Q4_K_M**, không phải F16 (máy không có F16). Activation đã qua lossy → weight importance lệch — có thể giải thích effect 0.24 thay vì ~0.5 dự đoán. **Chưa loại trừ được** (cần F16 source).
5. Tensor không đổi ở Q3 cũng có: `blk.27.attn_v`, `output.weight`, `token_embd` (2 file Q3) — K-quant giữ type không đổi → idempotent, như Q4.

## Điều tra tiếp (chờ quyết định — không tự ý làm)

- Loại trừ (4): chuẩn bị F16 của Hermes → build imatrix từ F16 → lặp lại gate Q3.
- Sửa (3): build lại imatrix bảo đảm `token_embd` có entry → gate Q3 lần 2.
- Cân nhắc lại ngưỡng: Q3_K_M mixed-precision đã tự đẩy tensor nhạy cảm lên type cao hơn → effect kỳ vọng 0.5 có thể đã được đẩy xuống 0.2–0.3.
- **Không** mở L1b (per-tensor search) cho tới khi một trong các nhánh trên sáng tỏ.

## Sự cố trong quá trình chạy

- **Attempt 1** (2026-10-20 18:57): session opencode đứt lúc stage 4 mới 30s → bash script bị kill; stage 1–3 (2 imatrix + 4 quantize Q4) đã hoàn thành, còn nguyên trên đĩa. Một `llama-perplexity` sót lại kẹt 14h (R nhưng không tiến triển) → kill -9 lúc 09:27.
- **Attempt 2** (09:29): vá script trước khi relaunch — counter `FAILED_RUNS` + trạng thái cuối trung thực; chạy `setsid nohup` (sống sót session đứt); resume-guard bỏ qua stages đã xong. Hoàn tất 10:26, `ALL L1A STAGES DONE` thật (0 fail).
- Report script ban đầu verdict #1 bị ngược ngưỡng (in FAIL cho Δ=0.0000) — đã vá (`invert=True`), bản đã lưu kèm thư mục này.

## Cách đọc lại số

```bash
python3 l1a_results/l1a_report.py        # in lại 5 con số + gate từ 8 JSON
grep '!!! PPL RUN FAILED' l1a_results/l1a_run_v2.log   # phải = 0
```
