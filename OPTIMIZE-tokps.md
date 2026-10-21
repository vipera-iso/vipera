# Plan tối ưu tok/s cho Vipera — 2026-10-21

> Mục tiêu: **decode tok/s nhanh nhất có thể** trên máy này, không phải trên giấy.
> Mọi con số dưới đây hoặc là (a) **đã đo**, ghi rõ nguồn, hoặc (b) **tính từ
> công thức có test**, hoặc (c) **target cần đo** — không có loại "suy đoán".
> Quy tắc cũ giữ nguyên: **số đo thắng văn bản**; ngưỡng gate không đổi sau
> khi thấy số.

---

## Cập nhật 2026-10-22 — đã sửa code, đã đo lại

Bốn việc dưới đây **đã vào `src/`** (kèm test) và số đo nằm ở
`tools/gpu_results/REPORT.md` §G:

| # | Thay đổi | Trước → sau | Bằng chứng |
|---|---|---|---|
| 1 | `cpu_threads = cores` (bỏ `cores - 1`) | +9% decode (ngl 10, q8_0), +19–21% ở f16 KV | §G2, median-3 |
| 2 | Reserve VRAM theo card (`max(0.25, min(1.0, 0.07×vram_total))` + 0.9 GB khi FA off) thay cho 1.0 + 0.3 cố định | planner: 10 → 14 layer (Q4 8K q8_0); 8 → 13 (16K) | §G1 (14 load / 15 fail), §G3 |
| 3 | **Placement ladder** trong `engine`/`server_backend`: `n, n-1, n-2, n/2` rồi mới CPU-only | hỏi 15 layer → load 13 (4.4 s), không còn rơi về 0 layer | §G4 |
| 4 | P0.4 **mlock**: nâng `RLIMIT_MEMLOCK` bằng root rồi đo | 6.26 vs 6.25 t/s → **+0.2%, dưới gate 5% ⇒ mlock không phải đòn bẩy trên máy này** | §G5 |
| 5 | Bỏ `meta["layer_bytes"]` khỏi budget: luôn dùng `size_bytes / n_layers` | Q3: planner hỏi 20 layer (trần 17) → rơi CPU; nay hỏi 17 → load ngay | §G7 |
| 6 | P1.2 **Q3 làm model chính**: hardlink + registry alias `hermes-4-14b-q3km-imatrix`, set `default_model` | **17/40 layer, 6.57 t/s** @8K q8_0 (Q4 control 6.18) | §G6, median-3 |
| 7 | P1.5 **`n_ubatch` 256/1024/2048** cho prefill | 512 → TTFT 4.52 s; 1024/2048 không load nổi ở 14 layer; cùng 12 layer thì TTFT trùng (4.659 vs 4.658 s) ⇒ **không có lợi ích, giữ 512** | §G8 |

Kết quả pipeline thật (`vipera bench run --ctx 8192`, median-3): **4.65 → 6.25
t/s**, TTFT 2.95 → 2.76 s, 10/40 → 14/40 layer. A/B chen kẽ trong cùng process
(loại trừ drift nhiệt ~8%/batch): **4.655 → 5.762 t/s = +24%**. 16K: 5.48 t/s
(trước: planner chọn 8 layer và thua CPU-only).

Phát hiện thêm khi chuyển model (lý do phải sửa planner lần nữa): **một lần tạo
context thất bại không trả lại VRAM** — sau 3 lần fail trên card 4 GB, rung 10
layer bình thường cũng không load nổi, nên "hỏi quá" nguy hiểm hơn "hỏi thiếu".
Vì vậy rung cuối của ladder là `n/4`, và budget dùng `size/n_layers` (không dùng
`layer_bytes` — nó dự đoán cao hơn trần thật 3–6 layer).

Còn lại của plan: P2.1 build `llama-server` với CUDA (máy không có `nvcc`),
P2.3 `--spec-type ngram-*`, P3.2 tách lane 64K. P1.5 coi như **đóng bằng số đo
âm tính** (giữ `n_batch`/`n_ubatch` 512).

**2026-10-23 — kết quả đọc tài liệu:** xem `OPTIMIZE-techniques.md`. Kết luận
ngắn: cửa lớn nhất còn lại là **MoE expert offload** (`-ot exps=CPU` /
`--n-cpu-moe`) cho LFM2.5 — 91% file là expert weights, và `llama-cpp-python`
**không** có tham số đó (chỉ `llama-server`). Cùng blocker `nvcc` với P2.3. Ba
đòn bẩy nữa đã đo và **loại**: `--no-op-offload` (TTFT +18%), `--no-mmap`,
`n_ubatch` lớn.

---

## 0. Phạm vi đọc

| Nhóm | Đã đọc |
|---|---|
| Code | `src/vipera/{planner,engine,config,hardware,server_backend,server}.py`, `commands/{bench,serve,system,model}_cmd.py` |
| Docs | `README.md`, `docs/benchmarks.md`, `docs/architecture.md` (§Known gaps), `docs/design/llama-server-backend.md`, `docs/configuration.md` |
| Backlog & số đo cũ | `OPTIMIZE-hermes4.md`, `fullcode.md` (đề xuất v2 + mục "Kết quả đo — 2026-10-21"), `tools/{layer_results,l1a2_results,l1a_results}/REPORT.md` |
| Tài liệu bên ngoài | `~/Downloads/vipera-tai-lieu-co-gia-tri/{A,B,S,S-mo-rong}` (AWQ/H2O/StreamingLLM, QMoE/HAWQ, SolidAttention/HeteroSpec/WhichQuantization, `S-mo-rong/INDEX.md` = open-problems pack), `docs/references/{papers,stack}` |
| Hạng deploy thật | `~/.vipera/config.json`, `~/.vipera/registry.json`, `~/.hermes/config.yaml` (providers agnes / local-deepseek:8080 / vipera:8686), `~/.hermes/hermes-agent/agent/model_metadata.py` |

**Trạng thái hiện trạng (không phải đề xuất):** L5a (PPL baseline), L1a (imatrix
Q8_0→Q3/Q4), L2 (9 KV dtype + fallback scale theo dtype), L3 (backend `server`
opt-in), L5 (bảng A–E) **đã làm và đã đo**. L1b **đóng theo gate** (0.3812 <
0.4). L4 chưa làm (2 model không có tensor MTP; DFlash kẹt VRAM).

---

## 1. Sự thật nền — đã xác minh lại hôm nay

| # | Sự thật | Bằng chứng |
|---|---|---|
| F1 | **Máy:** 4C/8T (AVX512, AVX512_VNNI), 15 GiB **không swap**, `MemAvailable ≈ 12.6 GiB`; GTX 1650 **4.00 GiB** (3.64 GiB trống, process 2 MiB) | `nproc`, `free -h`, `nvidia-smi` |
| F2 | **Có 2 venv, chỉ 1 cái có CUDA.** `~/.venvs/vipera` (py3.12, editable `vipera-ai`) = llama-cpp-python 0.3.36 **+ `libggml-cuda.so.0.25.3`** → đây là bản `~/.local/bin/vipera` thật. `~/vipera/.venv` (py3.14) = **CPU-only** (không `libggml-cuda`, `ldd` chỉ `libggml-cpu`) → nơi chạy `pytest` và `tools/bench_layers.py` | `ldd`, `nvidia-smi` trong lúc load 0.5B: `n_gpu_layers=99` mà VRAM vẫn **2 MiB** |
| F3 | **`tools/bench_layers.py` tự ép `--profile cpu`** để so backend chứ không so phần cứng → **toàn bộ bảng A–E ngày 2026-10-21 là CPU-only**, chưa chạm GPU lần nào | `bench_layers.py:24-26` |
| F4 | **`llama-server` local `GGML_CUDA=OFF`** (0.7.0-dev, `GGML_NATIVE=ON`) → backend `server` hôm nay cũng CPU-only, không so fair với inprocess CUDA | `~/dev/llama.cpp/build/CMakeCache.txt` |
| F5 | **Config hiện tại không start được.** `ctx 65536 + kv q8_0` → planner ra `ram_estimate 13.38 GB > 12.60 available + 0.5` → `check_host_ram` **REFUSE** | chạy `planner.plan()` với config thật, hôm nay |
| F6 | Planner hiện tại (chạy thật): `ctx 8192/f16 → ngl 9/40, FA off, mlock ON`; `8192/q8_0 → 10/40, FA on`; `16384/f16 → 8/40`; `65536/q4_0 → 8/40` | cùng lệnh trên |
| F7 | **`cpu_threads = cores − 1 = 3`** — máy 4 core vật lý/8 luồng, planner tự bỏ 1 core | `planner.py:100` |
| F8 | **FA trên GPU không bao giờ được chọn với KV f16**: `hw.vram_total_gb > 4` với `vram_total = 4.00` → **False** | `planner.py` nhánh else |
| F9 | **`server_backend` không truyền mlock/spec/MoE.** Comment `server_backend.py:86` nói llama-server không có `--mlock` → **sai**: 0.7.0-dev có `-lm, --load-mode MODE` với giá trị `mlock` / `mmap+mlock`. llama-server 0.7.0-dev còn có `--spec-type {ngram-simple,ngram-map-k,ngram-cache,…}`, `-md`, `--n-cpu-moe` | `llama-server --help` |
| F10 | **Hermes agent bắt buộc ≥ 64K window** (`MINIMUM_CONTEXT_LENGTH = 64_000`, `agent/model_metadata.py:272`, không env override; `context_switch_guard` + `cli_info_mixin` đều reject model < 64K) | grep code hermes |
| F11 | Số đã đo của các "lane" khác @64K: **LFM2.5-8B-A1B (hybrid KV) = 15.5 t/s**; **DeepSeek-V2-Lite @64K = 5.5 t/s** với prefix thật (13.85 t/s @20 tok) — MLA attention O(prefix); MoE + `mlock` **16.0 → 22.9 t/s**; Hermes-4 @64K = **0.13 t/s** (q4_0) / OOM (q8_0) | `OPTIMIZE-hermes4.md`, `docs/benchmarks.md`, comment `~/.hermes/config.yaml` |
| F12 | Best Hermes-4 đã đo: **4.00 t/s @8K f16 FA-off** (10/40), 2.86 @8K q8_0, prefill 7.4–10.1 s / ~470 tok | `docs/benchmarks.md` |
| F13 | Q3_K_M+imatrix đã có file thật `q8src-imbf-q3km.gguf` (7.32 GB): CPU-only **−25% load, −27% prefill, +3% decode (nhiễu), PPL không giảm** | `tools/layer_results/REPORT.md` §E |
| F14 | Gate L1b = 0.4 PPL, đo được 0.3812 → **đóng**. Không mở lại bằng cách sửa ngưỡng | `fullcode.md` §L1a v3 |
| F15 | 167 test pass, `ruff` clean; `fullcode.md` phải regen kèm phần đề xuất sau fence | `pytest -q` hôm nay |
| F16 | **`mlock` bị kernel chặn trên máy này.** `ulimit -l` = soft 8192 KB / hard 8192 KB, không nâng được khi không có root. Cả `-lm mlock` (llama-server 0.7.0-dev) lẫn `use_mlock=True` (llama-cpp-python) đều in `warning: failed to mlock … Cannot allocate memory` rồi **chạy tiếp như mmap thuần** | test thật với 0.5B, 2026-10-21 |
| F17 | Vì F16: planner `use_mlock=True` ở 8K là **no-op lặng lẽ**, và hàng "CPU only, `-lm mlock` 22.9 t/s" trong `docs/benchmarks.md` **không thể** đến từ mlock đang hoạt động trên máy này — trừ khi lúc đo `RLIMIT_MEMLOCK` đã được nâng (cần root). Cần đối chiếu lại trước khi trích dẫn | `docs/benchmarks.md` §Results |

---

## 2. Bottleneck thật (phân tích, có con số)

### 2.1 Trần tốc độ decode @8K — còn bao nhiêu chỗ?

Model 14B **dense** Q4_K_M = 9.00 GB đọc **đầy đủ mỗi token** (khác MoE chỉ đọc
2.4B active → MoE 16–22 t/s là bài toán dễ hơn 4×, đừng so trực tiếp).

```
ngl = 9 (số planner ra hôm nay)  →  CPU giữ 9.00 − 9×0.225 ≈ 6.98 GB
thương lượng DRAM thực dụng ≈ 45–50 GB/s (LPDDR4x-4266 2 kênh)
trần CPU-side ≈ 50 / 6.98 ≈ 7.2 t/s   →   đo được 4.00 t/s ≈ 55–70% trần
```

⇒ **Không còn dư địa 2× bằng "tối ưu cờ CPU thuần"**. Phần còn lại nằm ở:
thêm layer lên GPU, Q3 (giảm byte CPU), speculative, và — quan trọng nhất —
**bài toán 64K**, nơi trần là *attention*, không phải *bandwidth*.

### 2.2 Ba chế độ, ba bottleneck khác nhau

| Chế độ | Nút thắt | Trần/đo được | Khoảng cách còn lại |
|---|---|---|---|
| **@8K, 14B dense** | DRAM bandwidth phía CPU | trần ~7.2, đo 4.00 | **+30–80%** (Q3 + ngl sweep + spec) |
| **@16K–32K** | planner over-offload → `Failed to create llama_context` → fallback CPU-only | planner chọn 8, thật ~6 (ở 16K f16) | **fix calibration = lấy lại ~1.5–2×** so với CPU-only |
| **@64K (bắt buộc vì agent)** | attention trên CPU (FA + KV lượng tử), O(prefix) | 0.13 t/s Hermes; 5.5 t/s DeepSeek MLA; 15.5 t/s hybrid | **không có flag nào cứu dense 14B @64K** → phải đổi lane |

### 2.3 Điều gì ĐÃ được hưởng rồi (đừng tính lại thành "tối ưu mới")

- `mlock` **đã bật** ở 8K (F6: `use_mlock=True` khi headroom 4.67 GB > 3 GB) →
  con số 4.00 t/s **đã gồm mlock**. Mlock chỉ *mất* ở ctx ≥ 32K.
- Q3 file đã có, L1a đã đo, L1b đã đóng gate.
- Prefix caching đã có sẵn trong llama-cpp-python 0.3.36 (`pyproject` floor).

---

## 3. Plan — P0 → P4

Mỗi mục: **việc → cách đo → ngưỡng chấp nhận → rủi ro.**
Mọi phép đo: **median 3 lần**, Δ > 5% mới coi là thật (nhiễu đã đo ~3–5%),
lưu JSON vào `tools/*_results/`, ghi **venv nào** chạy.

### P0 — Gỡ blocker + làm trung thực phép đo (≤ 2h, rủi ro 0)

> **Trạng thái 2026-10-21: P0.1–P0.4 đã làm xong, test pass.**

| ID | Việc | Cách đo / bằng chứng | Nghiệm thu |
|---|---|---|---|
| **P0.1** | ✅ **Sửa config để serve được.** `~/.vipera/config.json` đang `ctx 65536 + kv q8_0` → REFUSE (F5). (`DEFAULT_CONFIG` trong `config.py` **đã là 8192/q8_0** — cái tự sát chỉ là config trên máy này.) | `vipera system config ctx 8192` → planner ra `ngl 10/40, FA on, mlock requested, est 8.76 GB < 11.7 GB` → **OK** | Banner hiện `ngl>0/40`, RAM còn > 1.5 GB |
| **P0.2** | ✅ **`system doctor` báo binding có GPU backend hay không** (`llama_supports_gpu_offload()`), kèm hint cài lại khi phát hiện máy có GPU nhưng binding CPU-only | `~/.venvs/vipera/bin/vipera system doctor` → `gpu offload: compiled in (CUDA/Metal) PASS`; `~/vipera/.venv/bin/vipera system doctor` → `CPU-only build WARN` + hint | 3 test mới trong `tests/test_system_cmd.py` |
| **P0.3** | ✅ **Sửa `server_backend.py`**: comment sai "llama-server không có --mlock" → đúng là `-lm mlock` (probe `--help` cho `--load-mode`, fallback im lặng với build cũ). Smoke test thật: llama-server 0.7.0-dev nhận `-lm mlock`, `/health` OK | 4 test mới trong `tests/test_server_backend.py` | `pytest -q` xanh, ruff sạch |
| **P0.4** | **RLIMIT_MEMLOCK (F16) — cần quyết định có root.** Máy này `ulimit -l = 8 MB` (soft = hard), không nâng được khi không root → **mlock không hoạt động ở bất kỳ đường nào**. Muốn đo tác dụng thật của mlock phải nâng trước: `/etc/security/limits.conf` (`user soft/hard memlock unlimited`) hoặc chạy service với `LimitMEMLOCK=infinity`, rồi **đo lại** 8K có/không mlock | Sau khi nâng: A/B `use_mlock` True/False, median 3 | Chỉ ghi "mlock được X%" vào docs khi số đo chạy với limit đã nâng. Nếu không có root → ghi rõ "mlock bất khả dụng trên máy này" vào `docs/benchmarks.md` |

### P1 — Ép GPU thật (mục tiêu: 4.00 → 6–7 t/s @8K)

> **Trạng thái 2026-10-21: P1.1 + P1.3 đã xong** (tool mới `tools/sweep_gpu.py`,
> số trong `tools/gpu_results/REPORT.md`). Trục `-ngl` và `threads` đã **hết
> chỗ**: Q3 + `-ngl 12` + `threads 4` = **6.00 t/s (median-3)**, pipeline sản
> phẩm hôm nay = **5.16 t/s (median-3)**. Muốn lên 7 chỉ còn P2 (spec) hoặc
> `n_batch`.

> **Đây là tầng quan trọng nhất**: từ 2026-10-21 chưa phép đo nào chạm GTX 1650
> (F3, F4), tức là tầng lớn nhất của plan **chưa từng được đo**.

| ID | Việc | Cách đo | Nghiệm thu / rủi ro |
|---|---|---|---|
| **P1.1** | ✅ **Sweep `-ngl`.** Đã viết `tools/sweep_gpu.py` (fresh engine/row, sample peak VRAM/RSS, **từ chối chạy** khi binding không có GPU backend) thay vì ép `bench_layers.py` | Q4 + Q3, `ngl 0…16`, 8K f16, 8K q8_0, 16K q8_0 | **ngl max thật: Q4 = 10 (VRAM 3476), Q3 = 13 (3624/3730); `ngl 14+` → fallback về 0 layer hoặc FAIL kèm RSS 40–55 GB.** Planner đúng ±1 layer ở Q4, **thiếu 3 layer ở Q3**, thiếu 4 ở 16K. `_GPU_COMPUTE_RESERVE_GB` giữ `0.3` (baseline CUDA context đã đo 1114 MiB ≈ phần planner trừ `1.0 + 0.3`). **2026-10-22:** trần đó là của **f16 KV + FA off**; với `q8_0` KV (FA on) trần thật là **14 layer** (Q4) / **17 layer** (Q3) — xem §G1 trong report. Reserve 1.3 GB đã bị thay bằng công thức hiệu chuẩn |
| **P1.2** | **Đổi model serve chính sang `q8src-imbf-q3km.gguf` (7.32 GB)** rồi đo lại ngl sweep: layer nhỏ hơn (0.183 vs 0.225 GB) → **cùng VRAM nhét thêm 1–2 layer**, CPU đọc ít hơn 18% | so Q4 vs Q3 **trên GPU** (chưa ai đo — số cũ là CPU-only, F13) | **Đã đo trên GPU: Q3 13 layer / 5.03 t / RSS 7.4 GB vs Q4 10 layer / 4.97 / 9.2 GB.** Còn thiếu: đổi `default_model` + chạy lại PPL gate 0.4 |
| **P1.3** | ✅ **`cpu_threads` 3 → so 3 / 4 / 8** (F7) + `n_threads_batch` riêng cho prefill (chưa làm) | 3×64 token mỗi giá trị, 8K f16, cả CPU-only lẫn `-ngl 12` | **Δ = +21% (CPU-only 4.78 vs 3.95) và +19% (ngl 12: 6.00 vs 5.05), cả hai median-3.** `threads 8` tệ hơn `threads 4`. → sửa `planner.py` `cores − 1` → `cores` + test **(chưa làm)** |
| **P1.4** | **`-fa on` trên GPU với KV f16** ở 16K/32K/64K (F8: planner không bao giờ chọn `>4` → FA off) — trên GPU FA là đường **nhanh** và tiết VRAM compute buffer | `--fa on/off` × ctx | Giữ nếu ≥ +5% TTFT/decode; rủi ro: VRAM, đo bằng nvidia-smi |
| **P1.5** | **`n_batch`/`n_ubatch` cho prefill**: 256 / 512 / 1024 (TTFT 7.4–10.1 s @470 tok là chậm, agent loop trả tiền mỗi turn) | prefill t/s qua `vipera bench run --ctx …` | Target TTFT −20% mà decode không giảm >5% |
| **P1.6** | **mlock ở ctx lớn** (`-lm mlock` / `use_mlock`) — planner hiện **tắt** mlock khi headroom < 3 GB (32K/64K) | 32K và 64K có/không mlock | ⛔ **Bị chặn bởi F16**: `ulimit -l = 8 MiB` (soft = hard), không nâng được khi không root → cả `-lm mlock` lẫn `use_mlock=True` đều `failed to mlock … Cannot allocate memory`. Cần root nâng `RLIMIT_MEMLOCK` trước (P0.4), sau đó mới đo được |

### P2 — Backend `server` + lỗ hổng cờ (mục tiêu: fair-compare + spec "miễn phí")

| ID | Việc | Cách đo | Nghiệm thu |
|---|---|---|---|
| **P2.1** | **Build lại `llama-server` với `GGML_CUDA=ON`** (hiện OFF, F4) → so **inprocess(CUDA) vs server(CUDA)** cùng plan | `tools/bench_layers.py --profile auto --backend inprocess server` trên model 0.5B **và** Hermes | Nếu server ≥ inprocess −3% → **flip default `backend: "server"`** (đường lui = config, không cần revert) |
| **P2.2** | **Bổ sung cờ trong `server_command()`**: `-lm mlock` (sửa comment sai ở `server_backend.py:86`), `-ctk/-ctv` đủ 9 dtype, `--n-cpu-moe`/`-ot` cho MoE (**known-gap**: planner không emit cờ MoE — đọc được ở DeepSeek/LFM), `--parallel 1` | unit test argv (đề xuất L3 đã viết: fake `llama-server` 40 dòng) | test đối chiếu argv + 1 lần chạy thật |
| **P2.3** | **`--spec-type ngram-*` (speculative không cần model draft, không tốn VRAM)** — llama-server 0.7.0-dev có `ngram-simple/ngram-map-k/ngram-cache` (F9), **chưa ai đề cập** trong `docs/architecture.md` (chỉ nói DFlash/`-md`). Agent prompt lặp prefix cao → trường hợp tốt cho ngram | A/B `--spec-type none` vs `ngram-cache` vs `ngram-map-k`, 64–128 token, prompt agent thật | Giữ nếu decode ≥ +10% và chất lượng sinh không đổi (cùng token count / cùng output) |
| **P2.4** | **Draft cổ điển `-md`**: Hermes-4 arch **qwen3** → so `vocab size` với Qwen3-0.6B/0.4B GGUF; nếu khớp → đo `--spec-type draft-simple`; **DFlash vẫn blocked** (cần ≥6 GB VRAM, máy 4 GB — giữ nguyên kết luận cũ) | A/B có/không draft, ghi **acceptance rate** | Giữ nếu ≥ +25%; nếu < +10% → ghi "đã đo, bỏ" |
| **P2.5** | **Test cancel-giữa-stream** cho `server.py` lock + proxy flags (đã nằm trong test plan của đề xuất v2, mục L3) | `tests/test_server.py` | 167 → >167 test, không regression |

### P3 — Bài toán 64K: nơi quyết định "nhanh nhất" thật sự

> F10: agent **từ chối** mọi window < 64K ⇒ mọi profile 8K/16K chỉ phục vụ
> `vipera chat`/bench, **không** phục vụ agent loop. Vì vậy nếu mục tiêu là
> "agent nhanh", tầng này quan trọng hơn P1.

| ID | Việc | Số liệu nền | Nghiệm thu |
|---|---|---|---|
| **P3.1** | **Chốt nhận định trong docs:** 64K + dense 14B = **ranh giới phần cứng**, không phải bug planner (0.13 t/s; q8_0 OOM; f16 = 10 GB không thể) | `OPTIMIZE-hermes4.md`, bảng benchmarks | Bảng "64K wall" vào `docs/benchmarks.md` |
| **P3.2** | **Tách lane:**
(a) **Fast lane = model hybrid/MoE** cho agent loop: `LFM2.5-8B-A1B` (**15.5 t/s**, hybrid KV → 64K vừa RAM) hoặc `DeepSeek-V2-Lite` với lever stack đã đo **1.49×** (`GGML_CPU_REPACK_MOE=0`, `-fa on`, `-t 4`, `-ctk q4_0`) + note MLA O(prefix) → 5.5 t/s;
(b) **Quality lane = Hermes-4** chạy ở **8K** cho `vipera chat`/bench, hoặc 64K `q4_0` khi thật sự cần | hai instance `:8686` / `:8687` (đã dự phòng ở mục 4 của `OPTIMIZE-hermes4.md`) | agent loop ≥ 5 t/s; RAM tính trước: 2 model mmap + 2 KV, **không swap** |
| **P3.3** | **(Nếu bắt buộc Hermes @64K)** các đường còn mở theo thứ tự: `q4_0 + FA on` chấp nhận 0.13 → **đổi model có MTP head** (`--spec-type draft-mtp`, llama.cpp hỗ trợ sẵn; **2 model hiện tại không có tensor MTP** — đã verify) | đo | Chỉ làm khi (a) bị chặn bởi 64K thật và (b) chấp nhận đổi model |
| **P3.4** | **Không làm** (đã loại, giữ nguyên): swap/zram (KV thrash), f16@64K (10 GB OOM), hạ ctx < 64K (agent reject), Q3-để-cứu-64K (nút thắt là attention, không phải bandwidth), StreamingLLM/H2O (**llama.cpp không hỗ trợ sliding-window KV** — sách `docs/references/papers/streamingllm…` chỉ dùng khi quyết định tự viết code research) | — | — |

### P4 — Chất lượng / giữ nguyên

- **L1b giữ đóng**: gate 0.4, đo 0.3812 (F14). Muốn mở → **tăng eval set** (ghi
  rõ là đo lại), không đổi ngưỡng.
- **L1a Q3** đã có số: dùng làm model chính của P1.2 nếu đo GPU thắng.
- **L5**: `bench run` chỉ đo speed; perplexity/KL per-tensor là code mới → chỉ
  viết khi một quyết định ở P1/P4 **cần** nó để chấp nhận.

---

## 4. Thứ tự thực thi + cổng đo

```
Tuần này
  1. P0.1 → P0.2 → P0.3         [blocking: giờ mới serve được + đo được trung thực]
  2. P1.1 (ngl sweep, tool mới)  [cho biết trần GPU thật]
  3. P1.3 threads → P1.5 batch → P1.4 FA-on-GPU   [cờ rẻ, đo nhanh]
  4. P1.2 Q3 làm model chính      [chỉ khi P1.1 xác nhận ngl ≥ planner]
Sau đó
  5. P2.1 build CUDA → P2.2 cờ → P2.3 ngram spec → P2.4 draft
  6. P3.2 tách lane (nếu mục tiêu là agent loop — quyết ở bước 1)
Luôn kèm
  - pytest + ruff trước mỗi commit; regen fullcode.md (giữ đề xuất sau fence)
  - mọi số mới vào tools/*_results/REPORT.md, có dòng `env:` + `median of 3`
```

**Cổng chung (gate):** một mục chỉ được ghi vào README/benchmarks khi có
(a) JSON thô trong repo, (b) điều kiện chạy lặp lại được (lệnh copy-paste),
(c) Δ > 5% so control hoặc lý do chất lượng (PPL/KL).

---

## 5. Bảng target — **đã đo (2026-10-21), số thật trong `tools/gpu_results/REPORT.md`**

| # | Config | Trước | Target | Đã đo | Trạng thái |
|---|---|---:|---:|---:|---|
| T1 | Hermes-4 Q4, 8K f16, env CUDA | **4.00 t/s** | **≥ 6.0** | **5.16** (pipeline `bench run`, `q8_0`+FA, median-3) / 4.97 (sweep f16) | ✗ **−14%**: `-ngl` (10 = max) và `threads` (4 = max) đã hết chỗ |
| T2 | Hermes-4 Q3+imatrix, 8K f16, GPU | chưa đo | **≥ 7.0** | **6.00** (ngl 12 + threads 4, median-3) | ✗ **−14%**: cần P2 (spec ngram) hoặc P1.5 (`n_batch`), không còn từ `-ngl`/`threads` |
| T3 | TTFT ~470 tok @8K | 7.4–10.1 s | **≤ 5.5 s** | **4.38–4.85 s** (ngl ≥ 12); pipeline 45 tok = 2.90 s | ✓ (n=1–3) |
| T4 | 16K | planner 8 → fail → CPU ~2 t/s | **≥ 4.5 t/s** | **4.87** (Q3, q8_0, ngl 12; planner chọn 8 → 4.55) | ✓ (n=1) — planner thiếu 4 layer |
| T5 | Agent loop @64K (Hermes) | 0.13 t/s | **không cải thiện được** → chuyển lane | 0.13 t/s (chưa đổi gì) | giữ — P3.1 |
| T6 | Agent loop @64K lane hybrid | LFM2.5 15.5 / DeepSeek 5.5 t/s | **giữ + đo lại** với cùng prompt agent thật | chưa đo | pending P3.2 |
| T7 | Spec ngram (server backend) | chưa có | **+10%** decode | chưa đo | pending P2.3 |

**Đòn bẩy đã hết ở tầng `-ngl`/`threads`:** để lên 7 t/s @8K chỉ còn
P2.3 (`--spec-type ngram-*`, không tốn VRAM) và P1.5 (`n_batch` cho prefill).
`threads = cores` (+19–21%, median-3) **đã được cộng vào** planner ngày
2026-10-22, cùng với reserve VRAM hiệu chuẩn và placement ladder — đó là
4.65 → 6.25 t/s, không phải một thí nghiệm riêng nữa.

---

## 6. Rủi ro & ràng buộc

1. **Không có swap** → mọi experiment RAM phải có preflight
   (`engine.check_host_ram`); Q3 + mlock ở 32K/64K là chỗ dễ chết nhất.
2. **2 venv lệch nhau** (F2) → số CPU và số GPU không được so với nhau; mọi
   report phải ghi env. Đây là lỗi methodology đã tồn tại trong bảng A–E.
3. **Đừng đụng gate L1b** và **đừng đổi ngưỡng sau khi thấy số** (F14).
4. **Đừng revert** vá CPU-only branch của planner (giữ quantized KV + FA) —
   `OPTIMIZE-hermes4.md` ghi rõ.
5. Backend flip default → cần đường lui bằng config (`backend: inprocess`) và
   test cancel-giữa-stream trước.
6. `fullcode.md`: regen mỗi lần đổi `src/`, **giữ nguyên phần đề xuất sau
   closing fence**.
