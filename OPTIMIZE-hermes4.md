# Tối ưu Hermes-4-14B chạy mượt trên máy này

> Created: 2026-10-19 · Symptom: **0.2 tok/s decode** (cần target ≥ 2-3 tok/s)
> Blocker chính: **flash attention trên CPU quá chậm** + planner ước tính VRAM sai nên không bao giờ offload GPU.

## Máy đích

- CPU: i5-11300H (4C/8T, AVX2 + AVX512F) · RAM 15GB · **không có swap**
- GPU: GTX 1650 4GB (~3.7GB trống, 380MB đã dùng)
- llama-cpp-python 0.3.36 — **đã compile CUDA** (quan sát được buffer `CUDA_Host` khi load)

## Triệu chứng đã đo (2026-10-19)

| Cấu hình | Decode | Ghi chú |
|---|---|---|
| Hermes-4 @65536, kv **q4_0** + FA ON, CPU-only | **0.1 tok/s** | 20 token mất 205s; 6 token mất 94s |
| Hermes-4 @65536, kv **q8_0** + FA ON, CPU-only | **0.2 tok/s** | 20 token mất 129s; 6 token mất 27.5s |
| LFM2.5-8B-A1B (hybrid attention, kv f16, FA OFF) | 15.5 tok/s | tham chiếu — KV 64K nhỏ nên f16 vẫn vừa RAM |
| Prompt eval Hermes-4 (FA ON) | ~29 t/s | prompt dài vài nghìn token = ~2-3 phút chỉ để nạp |

Lý thuyết CPU không-FA cho 14B Q4 (9GB / ~45GB/s bus) ≈ 4-5 tok/s → **FA+KV lượng tử chậm hơn ~50 lần**.
Khoảng cách q4_0 → q8_0 chỉ ~2x → bottleneck là **đường FA trên CPU**, không phải lượng tử KV.

## Chuỗi ràng buộc (không được bỏ sót khi fix)

1. **Hermes Agent bắt buộc ctx ≥ 64,000** — hằng số hard-code
   `MINIMUM_CONTEXT_LENGTH = 64_000` trong `hermes-agent/agent/model_metadata.py:272`,
   **không có env override**, áp dụng cả cho model chính lẫn auxiliary compression model.
   → không thể hạ ctx xuống 40960 (trained ctx của model) để tránh quant KV.
2. **llama.cpp BẮT BUỘC flash attention khi dùng KV cache lượng tử** — đã verify:
   `flash_attn=False + type_k=q8_0` → `ValueError: Failed to create llama_context`.
3. **KV @64K cho model này** (40 layer, 8 KV head, head_dim 128):
   - f16 = **10GB** → OOM tuyệt đối (máy 15GB, không swap)
   - q8_0 = **5.1GB** ← đang dùng
   - q4_0 = **2.5GB**
   Công thức: `2 × layers × kv_heads × head_dim × ctx × bytes/elem` (xem `planner._estimate_kv_cache_gb`).
4. **FA + KV lượng tử trên CPU = chậm kinh khủng** (số đo ở trên).
5. **Planner luôn ra `gpu_layers=0`** — BUG: nó trừ **toàn bộ** KV estimate vào VRAM
   (`usable_vram = safe_vram − kv_cache_gb`, `planner.py` ~dòng 97-107),
   trong khi llama.cpp thực tế đặt KV **per-layer** (layer nào trên GPU thì KV của layer đó trong VRAM).
   Với q4_0: `2.7 − 2.5 = 0.2GB < 0.225GB/layer` → 0 layer → rơi vào nhánh CPU-only.
6. Nhánh CPU-only **vừa được vá** (2026-10-19): trước đây nó ép `kv_cache="f16"` +
   `flash_attn=False` → 10GB KV → OOM. Giờ nó tôn trọng kv quant + bật FA
   (`flash_attn = kv_cache != "f16"`). Test: 20/20 pass trong `tests/test_planner.py`.
   **Đừng revert vá này.**

## Hướng fix (xếp theo ưu tiên)

### 1. Sửa ước tính VRAM trong planner → hybrid GPU offload (nên làm trước)

Tính `gpu_layers` bằng công thức per-layer, không trừ cả đống KV:

```
per_layer_kv_gb = kv_cache_gb / n_layers
gpu_layers ≈ (safe_vram − ??) / layer_gb   # giải pt: weights_gpu + kv_gpu ≤ safe_vram
   → gpu_layers ≤ (safe_vram) / (layer_gb + per_layer_kv_gb)
```

Dự kiến với q4_0 @65536: `2.7 / (0.225 + 0.0625) ≈ 9-12 layer` lên GPU
(weights ~2.5GB + KV của các layer GPU ~0.75GB ≈ 3.3GB < 3.7GB VRAM).

Lưu ý:
- `offload_kqv` nên bật khi có GPU layer để attention KQV tính trên GPU.
- FA sẽ chạy **per-layer**: layer GPU nhanh, layer CPU vẫn chậm → Gain có hạn,
  PHẢI đo benchmark thật trước khi mừng (dự kiến decode tăng vì attention giảm dần theo layer CPU).
- CUDA build sẵn trong llama-cpp-python rồi — không cần cài lại.

### 2. Test FA-ON-CPU bằng llama-server từ `~/dev/llama.cpp` (tách biệt nguyên nhân)

vipera hiện drive qua **llama-cpp-python** (bundled llama.cpp). Build local trong `~/dev/llama.cpp`
có thể mới hơn / tối ưu CPU FA tốt hơn. Test độc lập (không đụng vipera):

```bash
~/dev/llama.cpp/build/bin/llama-server -m ~/.vipera/models/NousResearch_Hermes-4-14B-Q4_K_M.gguf \
  -c 65536 -ctk q8_0 -ctv q8_0 -fa on -ngl 10 --port 8687
# rồi bench so sánh tok/s với instance vipera hiện tại
```

- Nếu `~/dev/llama.cpp` nhanh hơn đáng kể →考虑 cho vipera shell-out ra `llama-server`
  (README vipera nói "drives llama.cpp" nhưng engine đang dùng python lib — đổi sang subprocess
  là feature đúng hướng, cho phép dùng mọi build/flag).
- Nếu FA-CPU chậm như nhau → bottleneck là backend, chuyển hẳn hướng 1 + 3.

### 3. Fallback đã cân nhắc và LOẠI (đừng thử lại)

- **f16 + FA OFF @64K**: 10GB KV → không có swap → OOM chắc chắn. Không làm.
- **Swap/zram để chạy f16**: KV truy cập ngẫu nhiên 10GB → thrash, death. Không làm.
- **Hạ ctx xuống 40960 (trained)**: Hermes từ chối (< 64K). Không làm.
- **Quant Q3_K_M**: bottleneck là FA không phải bandwidth → gain ~20%, không đáng đổi độ chính xác.

### 4. (Nếu vẫn chậm) Chạy song song 2 instance

- `:8686` LFM2.5 (15.5 tok/s, provider nhanh cho agent loop thường xuyên)
- `:8687` Hermes-4 (chất lượng, chấp nhận chờ)
- Thêm provider thứ 2 trong `~/.hermes/config.yaml`. RAM sát: 2 model mmap (reclaimable)
  + 2 KV (LFM ~1-2GB + Hermes4 5.1GB) — cần test kỹ trước khi để lâu dài.

## Kết quả đo sau fix (2026-10-19, code trong repo)

| Cấu hình | GPU layers | TTFT | Decode | RAM lúc load |
|---|---|---|---|---|
| ctx **8192**, kv **q8_0**, FA ON | **11/40** | 4.8 s | **2.86 tok/s** (48 tok / 21.2 s) | 10.5 GB RSS, còn 3.9 GB |
| ctx **65536**, kv **q4_0**, FA ON | **9/40** | 5.0 s | **0.13 tok/s** (48 tok / 376.5 s) | 10.7 GB RSS, còn 1.9 GB |
| ctx **65536**, kv **q8_0**, FA ON | 7/40 | — | **OOM bị kill** (cần ~13.0 GB host, chỉ có 10.5 GB) | — |

Kết luận:

- Fix planner **có tác dụng**: 11 layer lên GPU và 2.86 tok/s ở 8K (từ 0.2 tok/s).
- **64K vẫn không đạt 2-3 tok/s trên máy này.** q4_0 64K chỉ 0.13 tok/s vì 31 layer
  còn lại chạy FA trên CPU; q8_0 64K không đủ RAM. Đây là giới hạn phần cứng
  (4 GB VRAM + 15 GB RAM không swap), không phải lỗi planner.
- `Engine` giờ **từ chối load trước** khi RAM không đủ (thay vì bị OOM-kill), và
  `use_mlock` chỉ bật khi còn >3 GB headroom sau khi trừ weights + KV host.
- Nếu bắt buộc 64K: dùng `kv_cache=q4_0` và đóng bớt ứng dụng; chấp nhận ~0.13 tok/s.
  Nếu cần tốc độ: dùng ctx 8K (2.86 tok/s) hoặc model nhỏ hơn cho agent loop.

## Checklist verify sau khi fix

```bash
# 1. Start đúng (LUÔN truyền --model — start trần có thể chọn nhầm model khác default_model)
vipera serve start --model NousResearch_Hermes-4-14B-Q4_K_M-2 --ctx 65536
#    → banner PHẢI có: layer offload > 0/40 (nếu fix planner), kv q8_0 [OK], flash attention [ON]

# 2. Benchmark — target ≥ 2-3 tok/s decode (hiện tại 0.2)
time curl -s http://127.0.0.1:8686/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"NousResearch_Hermes-4-14B-Q4_K_M-2","messages":[{"role":"user","content":"Say exactly: SPEED TEST"}],"max_tokens":20,"temperature":0}'

# 3. RAM an toàn sau load (available > 1GB, máy này không có swap)
free -h

# 4. Tests planner
cd ~/vipera && ~/.venvs/vipera/bin/python -m pytest tests/test_planner.py -q   # phải 16/16

# 5. End-to-end qua Hermes
hermes -z "Reply with exactly: OK" --provider vipera --model NousResearch_Hermes-4-14B-Q4_K_M-2
```

## Ghi chú vận hành

- **Đã có `vipera serve stop`** (pid file `~/.vipera/serve-<port>.pid`); dùng
  `vipera serve stop --force` nếu process không nhả sau 10 s.
- Start từ shell agent dễ bị SIGTERM oan → dùng `setsid nohup ... < /dev/null & disown`.
- Đổi model xong phải **refresh cache Model Picker của Hermes**: gõ `/model --refresh`
  trong TUI (file `~/.hermes/provider_models_cache.json` cache theo base_url, không tự biết
  vipera đã đổi model).
- Đổi `config.yaml` của Hermes → `hermes gateway restart`.
- Config hiện tại: `~/.vipera/config.json` → ctx 65536, kv q8_0, flash_attn true,
  default_model Hermes-4. **Lưu ý:** tổ hợp 64K + q8_0 cần ~13 GB host RAM; máy chỉ
  còn ~10.5 GB nên `Engine` sẽ từ chối load. Đổi sang `kv_cache=q4_0` nếu cần 64K,
  hoặc hạ ctx xuống 8192 để đạt ~2.9 tok/s. Backup config Hermes trước khi sửa: `~/.hermes/config.yaml.pre-vipera.bak`.

## Đo 2026-10-20: KV `f16` + flash attention OFF (ctx 8K/16K)

Máy: GTX 1650 4GB (3.7GB trống), 15GB RAM, không swap. Lệnh:
`vipera bench run NousResearch_Hermes-4-14B-Q4_K_M-2 --ctx 8192,16384 --max-tokens 64`
với `kv_cache=f16` trong config (planner tự đặt `flash_attn=False` khi KV là f16).

| Cấu hình | GPU layers thực tế | TTFT | Prefill | Decode | Tổng |
|---|---|---|---|---|---|
| 8K, kv **f16**, FA **off** | 10/40 | 3.2 s | 14.2 t/s | **4.00 t/s** | 18.9 s |
| 8K, kv q8_0, FA ON (mốc cũ) | 11/40 | 4.8 s | — | 2.86 t/s | 21.2 s (48 tok) |
| 16K, kv f16, FA off | **0/40** (GPU load fail → CPU) | 9.4 s | 4.8 t/s | 1.96 t/s | 41.6 s |

Kết luận:

- **f16 + FA off ở 8K nhanh hơn ~40%** so với q8_0 + FA on (4.00 vs 2.86 t/s):
  xác nhận nút cổ chai là đường flash-attention trên CPU, không phải lượng tử KV.
  Nên dùng `kv_cache=f16` khi KV vừa RAM; chỉ hạ xuống q8_0/q4_0 khi ctx lớn.
- **16K vẫn không offload được GPU** dù planner đã trừ thêm 0.3GB compute buffer:
  llama.cpp báo `ValueError: Failed to create llama_context`. Đo trực tiếp:
  `ngl=6` load được, `ngl=8` fail → lượng VRAM thực tế ở 16K lớn hơn mô hình
  "KV per-layer" của planner (xem mục Known gaps trong `docs/architecture.md`).
  Ở 16K nên chấp nhận CPU-only hoặc dùng model nhỏ hơn.
- **Prefix KV cache đã có sẵn trong llama-cpp-python 0.3.36** (không cần code
  trong Engine): `Llama.generate` tự tìm common prefix và chỉ eval phần mới.
  Đo trên qwen2.5-0.5b: prompt 1230 token mất 1.5 s lần đầu và 0.18 s cho lượt
  nối tiếp (+19 token). Vì vậy `pyproject.toml` nâng floor lên `>=0.3.36`.
