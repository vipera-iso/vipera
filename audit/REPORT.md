# Kiểm thử lỗi toàn diện — Vipera 1.2.0

- **Ngày:** 2026-10-24
- **Commit nền:** `4644007` (nhánh hiện tại; `src/vipera/optimizer.py` và `tools/l1a2_run.sh` đang có thay đổi sẵn, không thuộc lần kiểm thử này)
- **Môi trường:** Python 3.14.8 trong `vipera/.venv`, Linux, không GPU, không nạp GGUF thật
- **Kết quả nền:** `392 passed`, `ruff check src tests` sạch
- **Kết quả audit:** 20/20 probe **thất bại** — mỗi thất bại là một lỗi đã tái hiện được (xem cách chạy bên dưới)
- **Cập nhật sau khi sửa:** `418 passed` (1 skipped), `ruff check src tests` sạch; 11/20 probe đã chuyển sang pass, 9 probe còn lại vẫn fail đúng như mô tả
- **Cập nhật 2026-10-26:** 6 probe còn đỏ đã được sửa; `pytest audit` **20/20 pass**, bộ chính **517 passed** (1 skipped, 8 xfailed), `ruff` + `mypy` sạch. Các hồi quy tương ứng đã được đưa vào `tests/unit/` để CI giữ chúng xanh.

## Cách tái hiện

```bash
cd vipera
.venv/bin/python -m pytest -q          # 418 passed: bộ test gốc + hồi quy vẫn xanh
.venv/bin/python -m pytest audit -q    # 9 failed: các lỗi #4–#7, #9, #14, #15 còn mở
```

Thư mục `audit/` nằm ngoài `testpaths` nên không ảnh hưởng CI. Mỗi test phát biểu
hành vi **đúng**; nó fail vì code chưa làm đúng, không phải vì test hỏng.

## Tổng hợp

| # | Mức | Khu vực | Lỗi | Probe | Trạng thái |
|---|---|---|---|---|---|
| 1 | **Cao** | pool / server | `max_loaded` đóng engine **đang sinh token** của request khác | `test_pool_does_not_close_an_engine_that_is_mid_generation` | **Đã sửa** |
| 2 | **Cao** | pool / server | Load model lười chạy **trên event loop**, chặn toàn bộ HTTP | `test_pool_lazy_load_does_not_block_the_event_loop` | **Đã sửa** |
| 3 | **Cao** | optimize / shaping | Shaping **bỏ qua tool schema** → prompt vượt context, llama.cpp tự cắt | `test_shaping_accounts_for_the_tool_schema` | **Đã sửa** |
| 4 | Trung bình | optimize / shaping | Cắt history **làm mồ côi message `tool`** | `test_shaping_never_leaves_a_tool_message_without_its_assistant_turn` | **Đã sửa** |
| 5 | Trung bình | optimize / tuning | Phát lại split đo ở ctx 512 cho phiên ctx 8192 | `test_tuning_profile_is_not_replayed_at_a_larger_context` | Còn mở |
| 6 | Trung bình | protocols | Request sai **vẫn nạp model** trước khi trả 400 | `test_invalid_request_is_rejected_before_the_pool_loads_a_model` | Còn mở |
| 7 | Trung bình | protocols | Lỗi engine → **500 `text/plain`**, không phải JSON | `test_openai_engine_failure_returns_a_json_error_envelope`, `test_ollama_engine_failure_returns_its_error_shape` | **Đã sửa** |
| 8 | Trung bình | Ollama | `format` (JSON mode) bị **nhận rồi bỏ qua** | 2 test `..._format_json_...` | **Đã sửa** |
| 9 | Trung bình | serve | `serve start` thất bại **xoá pid file** của server đang chạy | `test_failed_serve_start_keeps_an_existing_pid_file` | **Đã sửa** |
| 10 | Thấp–TB | Ollama | `messages: []` được chuyển thẳng vào engine thay vì 400 | `test_ollama_chat_rejects_empty_messages` | **Đã sửa** |
| 11 | Thấp | Ollama | Non-stream trả `eval_count`/`prompt_eval_count` = 0 khi backend không có usage | `test_ollama_non_stream_counts_tokens_when_backend_reports_no_usage` | **Đã sửa** |
| 12 | Thấp | Ollama | `images` bị **nhận rồi bỏ qua** → trả lời như chưa thấy ảnh | `test_ollama_chat_does_not_silently_discard_images` | **Đã sửa** |
| 13 | Thấp | config / CLI / docs | `profile` trong config: `serve` đọc, `chat`/`bench` bỏ qua, docs nói không đọc | `test_chat_honours_the_configured_profile`, `test_bench_honours_the_configured_profile` | **Đã sửa** |
| 14 | Thấp | config | Không validate `profile`, `port`; bool sai **im lặng thành False** | 3 test `test_config_...` | **Đã sửa** |
| 15 | Thấp | planner | `iq4_nl` tính 0.5 B/elem thay vì 0.5625 (18 B/32) | `test_iq4_nl_kv_estimate_uses_the_exact_block_size` | **Đã sửa** |

### Đã sửa gì

- **#1 + #2 — pool thuê engine, nạp ngoài event loop.** `ModelPool.acquire()`
  đếm tham chiếu (`_in_use`), chỉ evict engine rảnh, và `wait()` khi mọi engine
  đang phục vụ; `Server` gọi `pool.acquire` qua `asyncio.to_thread`. Mọi route
  giữ `Lease` đến khi call trả về hoặc stream đóng (`protocols/streaming.py`),
  nên eviction không thể đóng context giữa lúc sinh token. Hồi quy:
  `tests/unit/test_pool.py::test_an_engine_serving_a_request_is_never_evicted`,
  `tests/unit/test_server.py::test_a_pool_load_does_not_run_on_the_event_loop`,
  `test_a_finished_stream_gives_its_pool_reference_back`.
- **#3 — tool schema vào ngân sách shaping.** `shape(..., tools=...)` trừ
  `estimate_tool_tokens(tools)` khỏi `prompt_budget`, cộng vào
  `estimated_before/after`; `OptimizeLayer.prepare` truyền `tools` xuống.
  Hồi quy: `test_a_tool_schema_is_charged_against_the_budget`,
  `test_the_tool_schema_is_part_of_the_estimated_prompt`,
  `test_the_tool_schema_reaches_the_shaper`.
- **#8 — Ollama `format`.** `"json"` → `{"type": "json_object"}`, object schema
  → `{"type": "json_object", "schema": ...}`, giá trị khác → 400 trước khi nạp
  model. Hồi quy: `test_ollama_chat_honours_format_json`,
  `test_ollama_chat_keeps_a_json_schema_format`,
  `test_ollama_rejects_an_unknown_format_before_loading_a_model`.
- **#10 — `messages: []`.** Trả 400 `{"error": ...}` trước khi resolve model.
- **#11 — token count non-stream.** `_nonstream_result` tính trong cùng worker
  thread: dùng `usage` nếu có, ngược lại `tokenize()` cho prompt và
  completion, nên `prompt_eval_count`/`eval_count` không còn là 0.
- **#12 — `images`.** Từ chối 400 thay vì trả lời như ảnh không tồn tại.
- **#13 — profile parity.** `chat_cmd._build` và `bench_cmd.run` truyền
  `profile=cfg.get("profile", "auto")`; docs (`README`, `docs/cli.md`,
  `docs/configuration.md`, `docs/architecture.md`) mô tả đúng nguồn cấu hình.
- **#14 (một phần) — validate `profile`.** `_coerce` sửa về `auto`,
  `_coerce_strict` từ chối giá trị lạ; `vipera system config profile turbo`
  giờ exit 1. `port` range và bool im lặng vẫn còn mở.

Các probe của #1, #2, #3, #8, #10–#13 giờ **pass**.

### Đợt sửa 2026-10-26

- **#4 — shaping bỏ theo nhóm.** Bước 2 xoá cả cặp `assistant(tool_calls)` +
  các `tool` đi liền sau, nên không còn message `tool` mồ côi. Hồi quy:
  `tests/unit/test_optimize_shaping.py::test_dropping_history_never_orphans_a_tool_result`.
- **#7 — lỗi engine là JSON 502.** `_backend_failure` của cả hai lớp protocol
  đã trả `502` với `{"error": ...}` (OpenAI: object có `message`/`type`;
  Ollama: chuỗi). Hai probe được cập nhật sang đúng hợp đồng 502 thay vì 500
  `text/plain`; `docs/http-api.md` mô tả lại hình dạng lỗi. Hồi quy đã có sẵn:
  `test_server.py::test_backend_failure_is_an_openai_shaped_502`,
  `test_protocols.py::test_ollama_backend_failure_uses_its_error_shape`.
- **#9 — pid file có chủ.** `start()` chỉ ghi/ xoá file của chính nó; nếu khởi
  động thất bại, nội dung cũ (pid của gateway đang chạy) được khôi phục. Hồi quy:
  `tests/unit/test_serve_cmd.py::test_failed_start_keeps_the_running_gateways_pid_file`.
- **#14 (phần `port`) — 1–65535.** `_coerce_strict` từ chối, `_coerce` sửa về
  mặc định kèm cảnh báo; `vipera system config port 99999` giờ exit 1 và không
  ghi file. Hồi quy: 2 test trong `tests/unit/test_config.py`.
- **#15 — `iq4_nl` = 18 B/32.** Hồi quy:
  `tests/unit/test_planner.py::test_iq4_nl_is_priced_at_its_ggml_block_size`.

Khi sửa #3, `_shrink` cũng được chỉnh để trừ độ dài marker khỏi `keep_chars`:
trước đó mỗi lần cắt còn để lại message dài hơn ngân sách đúng bằng marker
(`[... N characters omitted ...]`), nên request "đã cắt" vẫn vượt context vài
token.

---

## Chi tiết

### 1. [Cao] Eviction đóng engine khi engine đang sinh token

`ModelPool.get()` giữ `threading.Lock`, nạp engine mới rồi đóng engine cũ nhất
(`src/vipera/pool.py:70-84`). `server.build_app()` gọi `pool.get()` trong
`engine_for` (`src/vipera/server.py:38-40`), và route gọi `engine_for` **trước khi**
lấy `gateway.lock` (`src/vipera/protocols/openai.py:106`, `ollama.py:258`).

Vì vậy, với `max_loaded=1`, hai request đồng thời tới hai model:

1. request A đang chạy `engine.chat()` trong worker thread, đang giữ lock;
2. request B resolve model B → `pool.get("b")` đóng engine A **ngay lập tức**;
3. A vẫn đang dùng context llama.cpp đã bị giải phóng.

Probe đo được `closed_while_busy=True`. Với backend thật, đây là use-after-free
giữa lúc generation (crash/segfault), không chỉ là lỗi thống kê.

**Hướng sửa:** giữ tham chiếu đếm (refcount) cho engine đang phục vụ, hoặc chỉ
evict engine không nằm trong request hiện tại; ít nhất hãy resolve/evict **sau khi**
đã lấy `gateway.lock`.

### 2. [Cao] Nạp model lười chặn event loop

Cùng đường gọi trên: `pool.get()` → `factory` → `plan()` + `create_engine()` chạy
**đồng bộ trong event loop** của uvicorn. Với model 14B, đây là hàng chục giây đến
vài phút; `/health`, `/v1/models` và mọi request khác bị treo trong suốt thời gian
đó — trái với cam kết "health vẫn trả lời khi stream" mà chính repo kiểm thử
(`tests/unit/test_server.py::test_health_responsive_while_streaming`).

Probe: factory giả `sleep(0.4)`; một timer 50 ms bị trễ **0.40 s**.

**Hướng sửa:** `await asyncio.to_thread(pool.get, ...)` (kèm khóa theo alias để
không nạp trùng), hoặc nạp trước trong lúc khởi động. Lưu ý: khi đó phải đồng bộ
`loaded_ids()` và cách evict (xem #1).

### 3. [Cao] Shaping không tính tool schema

`OptimizeLayer.prepare(..., tools=...)` nhận `tools` nhưng không dùng
(`src/vipera/optimize/layer.py:37-49`); `shape()` chỉ nhìn `messages`. Trong khi đó
chat template render toàn bộ schema công cụ vào prompt.

Probe: messages 10 token + schema 120 000 ký tự (~40 000 token), context 512 →
`Shaping.applied == False`, `estimated_after == 10`. Thực tế llama.cpp sẽ cắt
prompt từ phía trước (đúng thứ shaping sinh ra để tránh).

Liên quan: `_MAX_PROMPT_CHARS` chỉ cộng `m.content`
(`openai.py:110`), nên `tools`/`tool_calls` lớn cũng bỏ qua giới hạn 500 KB.

**Hướng sửa:** ước lượng `tools` (và `tool_calls`) vào `estimate_tokens`/ngân sách;
trừ phần đó khỏi `prompt_budget` trước khi cắt history.

### 4. [Trung bình] Cắt history làm mồ côi message `tool`

Bước 2 xoá từng message một (`src/vipera/optimize/shaping.py:164-167`). Nếu xoá
`assistant(tool_calls)` nhưng giữ `tool`, chat template (Hermes/Qwen…) sẽ báo lỗi
"tool message without preceding tool_calls" — hoặc template bỏ qua kết quả tool.

Probe: với budget 494, kết quả còn
`[system, tool(c1), user("final")]` — `tool` không có assistant nào trước đó.

**Hướng sửa:** xoá theo nhóm `assistant(tool_calls) + các tool` đi liền nhau, hoặc
khi xoá assistant có `tool_calls` thì xoá luôn các `tool` phía sau.

### 5. [Trung bình] Tuning phát lại split ở context khác

`Tuning` lưu `ctx` (`src/vipera/optimize/tuning.py:99`) nhưng `apply_tuning()`
(`:155-190`) chỉ so `free_gb`, không so `p.context`. `vipera bench run` mặc định
`--ctx 512,2048`; split của lần đo 512 (thường nhanh nhất) sẽ được tái sử dụng khi
`serve` chạy ctx 8192/65536.

KV cache tỉ lệ với context: 20 layer vừa ở 512 có thể không vừa ở 8192. Hệ quả là
load fail rồi ladder tụt xuống (tốn thời gian, có thể về CPU-only), trong khi
profile trông như "đã áp dụng".

**Hướng sửa:** chỉ áp dụng khi `p.context <= profile.ctx` (context nhỏ hơn thì an
toàn), hoặc lưu/thu hồi split theo từng ctx; đồng thời dùng `hw` mới lúc nạp thay
vì `hw` chụp lúc khởi động trong pool mode.

### 6. [Trung bình] Request sai vẫn nạp model trước khi bị từ chối

`openai.py:106` resolve model **trước** các kiểm tra `_MAX_MESSAGES`,
`_MAX_PROMPT_CHARS`, `max_tokens`, `temperature`, `top_p` (`:107-116`). Với pool,
một request `max_tokens: 0` khiến model được nạp (có thể vài phút) rồi mới trả 400.
Probe xác nhận factory được gọi.

Cùng mẫu ở `/v1/embeddings` (`encoding_format`, embeddings-off) và
`/api/chat` (`_MAX_PROMPT_CHARS`), `/api/embed`.

**Hướng sửa:** validate toàn bộ body trước, chỉ `engine_for` sau cùng.

### 7. [Trung bình] Lỗi engine trả 500 `text/plain`

Không có exception handler nào trong `src/vipera/server.py`. Khi `engine.chat()`
raise (OOM, model lỗi, llama-server chết), Starlette trả
`500 text/plain "Internal Server Error"` cho cả `/v1/*` và `/api/*`. Client OpenAI
đọc `{"error": {...}}`, client Ollama đọc `{"error": "..."}`; cả hai đều không
parse được. Luồng streaming thì đã xử lý đúng (gửi SSE `{"error": ...}`), nên
không nhất quán.

Tài liệu hiện ghi `/v1` dùng `{"detail": ...}` của FastAPI (`docs/http-api.md:59-60`)
và Ollama trả `{"error": ...}` — 500 này vi phạm cả hai.

**Hướng sửa:** handler cho `BackendError`/`ViperaError` (và `Exception` cuối cùng)
theo từng protocol; ít nhất trả JSON có `detail`/`error`.

### 8. [Trung bình] Ollama bỏ qua `format` (JSON mode)

`OllamaChatRequest.format` và `OllamaGenerateRequest.format` được khai báo
(`src/vipera/protocols/ollama.py:83,93`) nhưng không được đọc ở `chat()`/`generate()`.
Client gửi `"format": "json"` (hoặc JSON schema) vẫn nhận văn xuôi tự do — sai
định dạng mà không có cảnh báo. Lớp OpenAI đã hỗ trợ `response_format`.

**Hướng sửa:** `"json"` → `{"type": "json_object"}`; dict → JSON schema
(`{"type": "json_schema", "json_schema": ...}`), truyền qua `kwargs["response_format"]`.

### 9. [Trung bình] `serve start` thất bại xoá pid file của server đang chạy

`start()` ghi pid file trước khi bind (`serve_cmd.py:347`) và `finally` luôn
`unlink` (`:362`). Nếu bind lỗi (port đang bận — thường gặp khi vô tình chạy hai
instance), pid file của instance **đang chạy** bị xoá; sau đó `vipera serve stop`
báo "OFFLINE" và không dừng được server.

Probe: pid file chứa `4242`; start thứ hai ném `OSError`; file biến mất.

**Hướng sửa:** chỉ xoá nếu nội dung vẫn là pid của chính tiến trình này, hoặc ghi
pid file sau khi bind thành công (uvicorn startup hook) và khôi phục bản cũ khi lỗi.

### 10. [Thấp–TB] Ollama nhận `messages: []`

`OllamaChatRequest.messages` không có `min_length` (`ollama.py:79-85`), khác với
`ChatRequest` của OpenAI (`Field(..., min_length=1)`). `_messages([])` → `[]` được
gửi vào engine; probe nhận **200** thay vì 400. Ollama thật trả 400 cho trường hợp
này.

### 11. [Thấp] Ollama non-stream trả token count bằng 0

Nhánh non-stream lấy thẳng `usage.get(...)` (`ollama.py:289-291` và `:327-329`),
trong khi nhánh streaming dùng `_counts()` có fallback `tokenize()` (`:512`).
`llama-cpp-python` 0.3.x không trả usage, nên client non-stream luôn thấy
`eval_count: 0` dù có text. Dùng `_counts(usage, text, active)` cho cả hai nhánh.

### 12. [Thấp] Ollama bỏ qua `images`

`images` được khai báo (`ollama.py:59`) nhưng `_messages()` (`:141-150`) chỉ lấy
`role/content/tool_calls`. Request gửi ảnh nhận **200** và model trả lời như thể
không có ảnh — nguy hiểm hơn một lỗi 400. Nên trả 400 ("model không hỗ trợ ảnh")
hoặc chuyển tiếp khi backend hỗ trợ vision.

### 13. [Thấp] `profile` trong config bị xử lý không nhất quán

- `serve_cmd.start` đọc `cfg["profile"]` (`serve_cmd.py:194`) và có test khẳng định.
- `chat_cmd._build` (`chat_cmd.py:38`) và `bench_cmd.run` (`bench_cmd.py:178`) gọi
  `plan()` **không** truyền `profile` → `profile: cpu` bị bỏ qua.
- Tài liệu lại nói ngược: `docs/cli.md:220` ("**not** read from config") và
  `docs/configuration.md:69` ("Read by: nothing currently").

**Hướng sửa:** chọn một nguồn sự thật — nên là "config được đọc ở mọi lệnh", thêm
`profile=cfg.get("profile", "auto")` cho chat/bench, và sửa docs.

### 14. [Thấp] Lỗ hổng validate config

- `profile` không được kiểm tra trong `_coerce_strict` (`config.py:153-182`):
  `system config profile turbo` ghi thành công, sau đó mọi `serve` exit vì
  `PlannerError`. Các enum khác (`kv_cache`, `backend`, `protocols`) đều bị chặn.
- `port` chỉ `int()` (`:71`), không giới hạn 1–65535; `99999` được lưu.
- `_coerce` xử lý bool bằng `str(v).lower() in ("1","true","yes","on")`
  (`:111-112`): `"ture"` → `False` **không cảnh báo**, trong khi int sai có cảnh báo
  (`:113-120`). Một typo có thể âm thầm tắt `shaping`/`tuning`/`embeddings`.

**Hướng sửa:** dùng `_parse_bool` (đã có) trong `_coerce` và cảnh báo khi sai;
validate `profile ∈ {auto,cpu,gpu,lowmem}` và `1 <= port <= 65535`.

### 15. [Thấp] `iq4_nl` ước lượng KV thấp hơn thực tế

`_KV_BYTES_PER_ELEM["iq4_nl"] = 0.5` (`planner.py:71`) trong khi ggml IQ4_NL là
18 B/32 weight = **0.5625**. Comment ngay trên nói các dtype mới dùng "exact ggml
block size". Sai 11% theo hướng **thiếu** → có thể offload dư một layer trên card
sát ngưỡng. (`q4_1`, `q8_0` cũng làm tròn nhưng được ghi là số đo hiệu chuẩn.)

---

## Đã đọc và kiểm tra, không thấy lỗi

Để tránh hiểu nhầm là chỉ soi một chỗ: các phần sau đã đọc hết và đối chiếu với
test hiện có, không phát hiện khiếm khuyết mới.

- **config**: quarantine file hỏng, ghi atomic, merge default, null-key fallback,
  chuẩn hoá `protocols`, strict parse.
- **registry**: đọc cả schema v2 và file phẳng 1.x, quarantine, ghi atomic.
- **planner**: budget CPU/GPU, KV theo head thật/GQA/hybrid, MoE `-ncmoe`, ladder
  dự phòng, mlock/RAM preflight, note trung thực sau fallback.
- **backends**: `server_command` (fit off, spec-type, `-lm`, embeddings), SSE
  parser, `_adopt_placement`, staged load, `Engine` mean-pooling, `close()` idempotent.
- **protocols**: SSE usage chỉ gửi một lần, `[DONE]`, tool-call delta, lock không
  nhả khi client ngắt, 404 model, `/api/tags` & `/api/ps`, tool arguments → object.
- **optimize**: prefix key/stats, shaping các bước 1/3, `_cut` head+tail, tắt layer.
- **sources**: router scheme, manifest Ollama 2 layout + host, hardlink/copy, HEAD URL.
- **CLI/UI**: REPL EOF/lỗi, `oneshot`, `serve stop` (pid stale/recycled), SIGTERM
  unwind, `bench` đếm reasoning token, inspect/quantize/imatrix sizing, wrap UI.

## Hạn chế của lần kiểm thử

- Dùng engine giả cho HTTP/pool; **không nạp GGUF thật**, không có GPU, nên các
  đường llama.cpp/llama-server (bao gồm use-after-free ở #1) chưa được tái hiện
  trên tiến trình thật.
- Không kiểm thử mạng (Hugging Face, URL, Ollama store thật).
- `mypy` không có trong `.venv` nên không chạy; `ruff` sạch.
- Các probe thời gian (#2) và luồng (#1) là tất định về mặt logic nhưng dùng
  `sleep` mô phỏng; trên máy rất chậm, ngưỡng có thể cần nới.

## Trạng thái

Cả 20 lỗi đã xong: #1–#15 (2026-10-24 và đợt 2026-10-26), `pytest audit` 20/20
pass và các hồi quy tương ứng đã nằm trong `tests/unit/` để CI giữ xanh.
