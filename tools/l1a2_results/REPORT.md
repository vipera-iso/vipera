# L1a v3 — imatrix đo từ nguồn Q8_0 (requantize thật)

**Ngày:** 2026-10-21 · **Trạng thái:** xong, 4/4 arm `rc=0` · **Kết quả:** Q4 trong nhiễu,
Q3 có hiệu ứng thật nhưng vẫn dưới ngưỡng gate đã chốt (0.3812 < 0.4).

---

## Vì sao phải chạy lại (khác L1a v2)

L1a v2 lượng tử hoá `Q4_K_M → Q4_K_M`. `llama-quantize` copy nguyên byte khi type
nguồn trùng type đích (`llama-quant.cpp`: `quantize = cur_type != new_type`), và
imatrix chỉ được tra **bên trong** nhánh lượng tử hoá thật. Nên arm Q4 của v2 là
no-op cấu trúc: `+0.0000` không phải "imatrix vô dụng ở Q4".

v3 sửa đúng chỗ đó: nguồn là **Q8_0** (15.7 GB, ~0.01% loss so với BF16) nên
`Q8_0 → Q4_K_M` và `Q8_0 → Q3_K_M` **bắt buộc** đi qua bộ lượng tử hoá, và imatrix
có đường tác động vào từng weight.

## Bằng chứng "imatrix có thật sự đi vào cuộc"

`tools/gguf_quant_audit.py` so từng tensor giữa hai output cùng nguồn:

| Cặp | Tensor đổi **type** | Tensor đổi **data** |
|---|---:|---:|
| `q8src-ctl-q4km` vs `q8src-imbf-q4km` | 0/443 | **280/443** |
| `q8src-ctl-q3km` vs `q8src-imbf-q3km` | 0/443 | **280/443** |
| (v2, để so) `ctl-hermes-q4km` vs `l1a-hermes-q4km` | 0/443 | 0/443 |

Cùng type (type do recipe quyết định), nhưng **280/443 tensor đổi giá trị** — đó
đúng là chỗ imatrix tác động. Ở v2 con số này là 0.

## Số đo

Điều kiện y hệt L5a/L1a v2: `llama-perplexity 0.7.0-dev (build 1, commit 797de63)`,
`ctx=512, ngl=0, chunks=24, n_seq=4, scored_tok=12264`, eval `wikitext_sample.txt`
(sha256 `2a8ef09474c9`), CPU-only, chạy `setsid` nên không chết theo session.

| Arm | PPL ± stderr | Δ so control | % | Đọc |
|---|---:|---:|---:|---|
| `Q4_K_M` không imatrix | 7.0056 ± 0.2256 | — | — | control |
| `Q4_K_M` + imatrix BF16 | **6.9383** ± 0.2238 | **−0.0673** | −0.96% | **0.3σ → nhiễu** |
| `Q3_K_M` không imatrix | 7.3617 ± 0.2407 | — | — | control |
| `Q3_K_M` + imatrix BF16 | **6.9805** ± 0.2253 | **−0.3812** | −5.18% | **1.6σ → thật, nhưng < ngưỡng 0.4** |

stderr là độ bất định của eval set (12,264 token), không phải nhiễu run-to-run:
run lặp lại cho **đúng từng số lẻ**. Vì vậy Δ nhỏ hơn 1σ thì không kết luận được
là có lợi trên văn bản khác; Δ = 1.6σ thì có lợi nhưng chưa đủ để tuyên bố.

## Đọc số

1. **Q4_K_M: imatrix gần như không đổi gì** (−0.96%, 0.3σ). Hợp lý — Q4_K_M đã
   mixed-precision, tensor nhạy cảm được đẩy lên type cao sẵn, còn ít chỗ để
   imatrix cải thiện.
2. **Q3_K_M: imatrix cải thiện 5.18%** — cùng chiều với v2 (−3.24%), khác nguồn
   đo và khác imatrix. Hai phép đo độc lập cùng dấu ⇒ hiệu ứng có thật; biên độ
   0.24–0.38 PPL.
3. **Kết quả thực dụng quan trọng nhất:**
   `Q3_K_M + imatrix` = **6.9805**, `Q4_K_M` (không imatrix) = **7.0056**: Δ = 0.025
   PPL = **0.11σ**, tức **tương đương trong sai số** — không phải "tốt hơn".
   Điều bảo vệ được là: chất lượng không giảm khi file **7.32 GB thay vì 9.00 GB
   (−1.68 GB, −18.7%)**, và perplexity chạy **nhanh hơn 27%** (23.6 vs 18.8 tok/s)
   trên cùng máy CPU-only. Đừng viết "Q3 tốt hơn Q4" ở README.
4. **Verdict theo gate đã chốt trước:** Q4 không kết luận được; Q3 = 0.3812 vẫn
   **dưới** ngưỡng 0.4 ⇒ **chưa mở L1b**. Gate đúng như đã viết — không sửa
   ngưỡng sau khi thấy số.
5. **Về việc sát ngưỡng:** 0.3812 chỉ cách 0.4 một chút, nên một eval set lớn hơn
   12k token rất có thể vượt ngưỡng. Quyết định: **giữ nguyên ngưỡng đã chốt
   trước khi thấy số và không mở L1b ở vòng này** — đổi ngưỡng sau khi nhìn kết
   quả thì gate không còn là gate. Muốn vượt thì chạy thêm eval set (ghi rõ là
   đo lại, không phải chỉnh ngưỡng).

## Hạn chế (phải nói kèm số)

- Một eval set duy nhất (WikiText, 12k token). Không đo trên văn bản khác, không
  đo KL per-tensor, không kiểm tra chất lượng hội thoại.
- Imatrix là bản có sẵn của bartowski build từ **BF16** của cùng model — không
  phải build tại máy này. Nó loại trừ được giả thuyết "imatrix từ Q4_K_M bị lệch"
  của v2, nhưng không chứng minh được "imatrix tự build cũng cho kết quả này".
- Q3_K_M ở 3-bit: 5% PPL là con số đẹp, nhưng 3-bit vẫn dễ hỏng ở suy luận
  nhiều bước; chưa đo task-level.

## Tái lập

```bash
bash tools/l1a2_run.sh          # resume: bỏ qua bước đã có, PPL đã có rc=0 thì skip
cat tools/l1a2_results/report_numbers.txt
python tools/gguf_quant_audit.py \
  --a ~/.vipera/models/q8src-ctl-q4km.gguf --b ~/.vipera/models/q8src-imbf-q4km.gguf
```
