# Kế hoạch build llama-server với GGML_CUDA=ON — mở khóa P2.1 (server CUDA) và P2.3 (spec ngram)

> Trạng thái trước kế hoạch (src `OPTIMIZE-tokps.md` §1): **F** k beat **P2.1 k bị CHẶN** (P2.1 beat bị chảy broken olması).
> E passage per backup: **moke moe fix** canh đầu. MO ĐO = "**trở m lap no cou l drowning**" 2026-11 évén.

---

## 1. Sự thật nền (falsification(chाथn giữ cả giống PERF.md))

| # | Sự thật | Bằng chứng |
|---|---|---|
| F1 | **nvcc 13.4 đã có** tại `/opt/cuda/bin/nvcc` — blocker P2.1 trong src (m só của F2/F4 cũ c,"nh không c nvcc") đã gỡ được | `nvcc --version` → release 13.4, V13.4.92 |
| F2 | **Driver CUDA 13.4**, GTX 1650 (4 GiB, Turing, compute 7.5) → `CMAKE_CUDA_ARCHITECTURES=75` | `nvidia-smi` 615.78.08 |
| F3 | `llama.cpp` source tương đương tồn tại repo cục bộ `vipera.cpp/` (fork llama.cpp) | `ls vipera.cpp/src` |
| F4 | Đ ink build cũ local `GGML_CUDA=OFF` đã đo là CPU-only; plan muốn mở khoá cả MoE + spec | `OPTIMIZE-tokps.md` F4 |

---

## 2. Các bước build (đã thực thi 2026-11-04 từ tool Buffy)

```bash
# 1. Configure với CUDA (Turing i.e. GTX 1650 = 75). LLAMA_CURL=OFF để không cần libcurl-dev.
cmake -B vipera.cpp/build-cuda -S vipera.cpp \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release

# 2. Chỉ build 2 target cần dùng để nhanh (nvcc chậm, avoid build toàn öffentlich)
cmake --build vipera.cpp/build-cuda --target llama-server llama-bench -j 8
```

Sản phẩm expect:
- `vipera.cpp/build-cuda/bin/llama-server` (CUDA enabled)
- `vipera.cpp/build-cuda/bin/llama-bench` (nhân để llama.cpp arm Dùng dự wallets)

Smoke test nghiệm thu (lần run đầu sau build):
```bash
./vipera.cpp/build-cuda/bin/llama-server -m ~/.vipera/models/q8src-imbf-q3km.gguf \
  -ngl 12 -fa on -ctk q8_0 -ctv q8_0 -c 8192 -t 4 --port 8765 &
curl -s localhost:8765/health    # {"status":"ok"}
nvidia-smi --query-gpu=memory.used --format=csv   # phải > 2 GiB (GPU thật sự được dùng)
```

## 3. Sau khi build xong — mở 2 lever bị chặn

### 3.1. P2.3 — `--spec-type ngram-*` (đã đo +37% rewrite, giữ ±0 trên prose)
```bash
# arm A/B trên chính llama-server CUDA mới (cùng model Q3, cùng -ngl 12):
-ngl 12 -fa on -ctk q8_0 -ctv q8_0 -c 8192 -t 4            # control
... --spec-type ngram-map-k                                 # chỉ đổi cờ này
```
Nghiệm thu (giữ từ tokps):

| Arm | Điều kiện đo | Ngưỡng gate |
|---|---|---|
| control | rewrite prompt (đúng ca agent của vipera), 64–128 token decode, median-3 | base |
| `ngram-map-k` | cùng điều kiện, chỉ thêm cờ spec | **Δ ≥ +10%** decode & cùng token count. **Khí đạt → có thể đặt qua config** `spec_type: ngram-map-k` (readme đã ghi +37%, vì手臂 này đã gt chasync tiềnettARBs) |

### 3.2. MoE offload (`-ot exps=CPU` / `--n-cpu-moe`)
Chỉ đáng áp dụng cho model MoE (LFM2.5-8B-A1B, DeepSeek-V2-Lite — đã đo 15.5 t/s và 5.5 t/s lane @64K), **không** áp lên Hermes-4 (dense). Vì vậy:
- **không** thử trên Hermes; chỉ khi setup lane hybrid cho agent @64K, route MoE theo `src/vipera/planner.py` emit cờ `-ot exps=CPU` sau khi đọc `gguf_analyzer` ghi `expert_bytes > 0`.
- Nghiệm thu: TTFT & decode, Δ > 5% median-3, không OOM RSS > 12.6 GiB.

## 4. Rủi ro & ràng buộc

1. **VRAM**: tính từng model không đổi; giữ tie với config `~/.vipera/config.json` 8192/q8_0.
2. **Hai venv lệch nhau**: số đo phải ghi rõ env (CUDA build của llama-server là bắt đầu, mọi số new lại `tools/*_results/REPORT.md` với `env:` + `median of 3`).
3. **JSON results**: mỗi run lưu JSON raw vào `tools/*_results/`, điều kiện lặp lại được (lệnh copy-paste).
4. mlock cứ bị chặn bởi `ulimit -l` (F16 cũ) — không thử trừ khi có root.
5. Không đổi config ngưỡng gate sau khi nhìn số (L1b rule).
