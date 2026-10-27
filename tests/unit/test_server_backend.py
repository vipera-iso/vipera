import json

import pytest

from vipera.backends import server as sb
from vipera.optimizer import OptimizerError
from vipera.planner import Plan


def _plan(**overrides):
    base = dict(
        gpu_layers=7,
        cpu_threads=3,
        context=8192,
        kv_cache="q5_0",
        flash_attn=True,
        n_batch=256,
        offload_kqv=True,
        use_mlock=True,
        note="Hybrid",
    )
    base.update(overrides)
    return Plan(**base)


class _Response:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


class _Payload:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _Proc:
    def __init__(self, alive=True):
        self.alive = alive
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False

    def wait(self, timeout=None):
        return 0


def _engine():
    """A ServerEngine without a child process or an HTTP port."""
    engine = sb.ServerEngine.__new__(sb.ServerEngine)
    engine._closed = False
    engine._proc = None
    engine._log = None
    engine._port = 1234
    engine._log_path = None
    return engine


def test_request_timeout_is_a_generous_backstop():
    """The bound exists to break a hang, not to cut off slow hardware."""

    assert sb._request_timeout(0) == sb._MIN_REQUEST_TIMEOUT_S
    assert sb._request_timeout(None) == sb._MIN_REQUEST_TIMEOUT_S
    assert sb._request_timeout("nonsense") == sb._MIN_REQUEST_TIMEOUT_S
    # A long generation gets proportionally longer than the floor.
    assert sb._request_timeout(8192) == pytest.approx(8192 / sb._MIN_TOKENS_PER_S)


def test_chat_bounds_each_child_request_with_a_timeout(monkeypatch):
    engine = _engine()
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["timeout"] = timeout
        return _Payload({"choices": []})

    monkeypatch.setattr(sb.urllib.request, "urlopen", fake_urlopen)

    engine.chat([{"role": "user", "content": "hi"}], max_tokens=4096)

    assert seen["timeout"] == sb._request_timeout(4096)


def test_health_reports_exited_when_the_child_is_gone():
    engine = _engine()
    assert engine.health() == "exited"  # no child was ever started

    engine._proc = _Proc(alive=False)
    assert engine.health() == "exited"  # the child died


def test_health_asks_the_child_and_reports_its_state(monkeypatch):
    engine = _engine()
    engine._proc = _Proc(alive=True)

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(sb.urllib.request, "urlopen", lambda url, timeout=None: _Ok())
    assert engine.health() == "ok"

    def _starting(url, timeout=None):
        raise sb.urllib.error.HTTPError(url, 503, "loading", {}, None)

    monkeypatch.setattr(sb.urllib.request, "urlopen", _starting)
    assert engine.health() == "starting"

    def _unreachable(url, timeout=None):
        raise sb.urllib.error.URLError("connection refused")

    monkeypatch.setattr(sb.urllib.request, "urlopen", _unreachable)
    assert engine.health() == "unreachable"


# --- plan -> flags -----------------------------------------------------------


def test_server_command_maps_the_plan_to_flags():
    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 9000)

    assert command[0] == "/bin/llama-server"
    assert command[command.index("-m") + 1] == "/m.gguf"
    assert command[command.index("--port") + 1] == "9000"
    assert command[command.index("-c") + 1] == "8192"
    assert command[command.index("-ngl") + 1] == "7"
    assert command[command.index("-t") + 1] == "3"
    assert command[command.index("-b") + 1] == "256"
    # KV quantization is passed for both halves of the cache.
    assert command[command.index("-ctk") + 1] == "q5_0"
    assert command[command.index("-ctv") + 1] == "q5_0"
    assert command[command.index("-fa") + 1] == "on"
    assert "--jinja" in command
    # A non-existent binary fails the --help probe, so the mlock flag is left
    # out: passing it to a build without --load-mode kills the child.
    assert "-lm" not in command


def test_server_command_emits_moe_flags_and_uses_full_ngl():
    command = sb.server_command(
        "/bin/llama-server",
        "/m.gguf",
        _plan(gpu_layers=14, n_layers=24, moe_cpu_layers=12, n_batch=512, n_ubatch=896),
        1,
    )

    # A MoE plan keeps all 24 blocks for dense tensors and moves the first 12
    # layers' routed experts to the CPU. -ngl is 25 because llama.cpp counts
    # the output layer as one group past block_count.
    assert command[command.index("-ngl") + 1] == "25"
    assert command[command.index("-ncmoe") + 1] == "12"
    # -ub is the prefill lever and -b must be at least as large.
    assert command[command.index("-ub") + 1] == "896"
    assert command[command.index("-b") + 1] == "896"


def test_server_command_passes_the_expert_pool_when_the_planner_asked_for_one():
    command = sb.server_command(
        "/bin/llama-server",
        "/m.gguf",
        _plan(gpu_layers=24, n_layers=24, moe_cpu_layers=24, moe_cache_mib=2048),
        1,
        moe_cache_supported=True,
    )

    assert command[command.index("--moe-cache-mib") + 1] == "2048"


def test_server_command_drops_the_expert_pool_on_builds_without_the_flag():
    command = sb.server_command(
        "/bin/llama-server",
        "/m.gguf",
        _plan(gpu_layers=24, n_layers=24, moe_cpu_layers=24, moe_cache_mib=2048),
        1,
        moe_cache_supported=False,
    )

    assert "--moe-cache-mib" not in command


def test_server_command_ignores_the_pool_without_cpu_experts():
    # Nothing lives in RAM for a dense plan, so a pool would only take VRAM.
    command = sb.server_command(
        "/bin/llama-server",
        "/m.gguf",
        _plan(moe_cache_mib=2048),
        1,
        moe_cache_supported=True,
    )

    assert "--moe-cache-mib" not in command


def test_supports_moe_cache_reads_the_binary_help(monkeypatch):
    class _Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""

    monkeypatch.setattr(
        sb.subprocess,
        "run",
        lambda *a, **kw: _Result("  --moe-cache-mib N  GPU cache size in MiB"),
    )
    assert sb.supports_moe_cache("/bin/llama-server") is True

    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **kw: _Result("no such option here"))
    assert sb.supports_moe_cache("/bin/llama-server") is False

    def _boom(*a, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(sb.subprocess, "run", _boom)
    assert sb.supports_moe_cache("/bin/llama-server") is False


def test_server_command_offloads_the_output_layer_with_a_full_plan():
    # A full plan means every block, so ask for the extra group llama.cpp uses
    # for the output layer: -ngl 24 on a 24-block model offloads 24 of 25 and
    # decoded 186.7 t/s against 203.7 t/s with 25 of 25 (qwen2.5-0.5B).
    full = sb.server_command("/bin/llama-server", "/m.gguf", _plan(gpu_layers=24, n_layers=24), 1)
    assert full[full.index("-ngl") + 1] == "25"

    # A partial plan keeps exactly the layers the planner counted.
    partial = sb.server_command(
        "/bin/llama-server", "/m.gguf", _plan(gpu_layers=17, n_layers=40), 1
    )
    assert partial[partial.index("-ngl") + 1] == "17"


def test_server_command_omits_ncmoe_for_dense_plans():
    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1)
    assert "-ncmoe" not in command


def test_server_command_passes_spec_type_when_supported():
    # Measured +37% decode on a copy-heavy prompt, -1% on fresh prose; the flag
    # is only safe on builds that list it in --help.
    command = sb.server_command(
        "/bin/llama-server",
        "/m.gguf",
        _plan(),
        1,
        spec_type="ngram-map-k",
        spec_type_supported=True,
    )

    assert command[command.index("--spec-type") + 1] == "ngram-map-k"


def test_server_command_skips_spec_type_when_unset_unsupported_or_none():
    for kwargs in (
        {"spec_type": None, "spec_type_supported": True},
        {"spec_type": "ngram-map-k", "spec_type_supported": False},
        {"spec_type": "none", "spec_type_supported": True},
    ):
        command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1, **kwargs)
        assert "--spec-type" not in command


def test_supports_spec_type_reads_the_binary_help(monkeypatch):
    class _Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""

    monkeypatch.setattr(
        sb.subprocess,
        "run",
        lambda *a, **kw: _Result("--spec-type none,ngram-simple,ngram-map-k"),
    )
    assert sb.supports_spec_type("/bin/llama-server") is True

    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **kw: _Result("no such option here"))
    assert sb.supports_spec_type("/bin/llama-server") is False

    def _boom(*a, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(sb.subprocess, "run", _boom)
    assert sb.supports_spec_type("/bin/llama-server") is False


def test_server_command_disables_auto_fit_when_supported():
    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1, fit_supported=True)
    assert command[command.index("-fit") + 1] == "off"

    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1, fit_supported=False)
    assert "-fit" not in command


def test_probe_devices_detects_cuda0(monkeypatch):
    class _Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result(
            "Available devices:\n  CUDA0: NVIDIA GeForce GTX 1650 (3730 MiB, 3674 MiB free)\n"
        )

    monkeypatch.setattr(sb.subprocess, "run", fake_run)

    output = sb.probe_devices("/bin/llama-server")

    assert calls == [["/bin/llama-server", "--list-devices"]]
    assert "CUDA0" in output
    assert sb.has_cuda_device(output) is True


def test_probe_devices_reports_cpu_only_and_probe_failure(monkeypatch):
    class _Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""

    monkeypatch.setattr(
        sb.subprocess,
        "run",
        lambda *a, **kw: _Result("Available devices:\n  (none)\n"),
    )

    output = sb.probe_devices("/bin/llama-server")

    assert "none" in output
    assert sb.has_cuda_device(output) is False

    def _boom(*a, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(sb.subprocess, "run", _boom)

    assert sb.probe_devices("/bin/llama-server") is None
    assert sb.has_cuda_device(None) is False


@pytest.mark.parametrize(
    "device",
    [
        "CUDA0: NVIDIA GeForce GTX 1650 (3730 MiB, 3600 MiB free)",
        "ROCm0: AMD Radeon RX 7900 XTX (24560 MiB, 24000 MiB free)",
        "HIP0: AMD Radeon RX 6800 (16384 MiB, 16000 MiB free)",
        "SYCL0: Intel(R) Arc(TM) A770 Graphics (16384 MiB, 16000 MiB free)",
        "Vulkan0: Intel(R) Arc(TM) A750 Graphics (8192 MiB, 8000 MiB free)",
        "MTL0: Apple M2 Pro (16384 MiB, 15000 MiB free)",
    ],
)
def test_any_vendor_device_counts_as_an_accelerator(device):
    devices = f"Available devices:\n  {device}\n"

    assert sb.has_accelerator_device(devices) is True
    assert sb.accelerator_device(devices) == device


def test_cpu_only_and_empty_device_lists_have_no_accelerator():
    assert sb.has_accelerator_device("Available devices:\n  (none)\n") is False
    assert sb.accelerator_device("Available devices:\n  (none)\n") is None
    assert sb.has_accelerator_device(None) is False


def test_supports_n_cpu_moe_reads_the_binary_help(monkeypatch):
    class _Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""

    monkeypatch.setattr(
        sb.subprocess,
        "run",
        lambda *a, **kw: _Result("  -ncmoe, --n-cpu-moe N  keep experts on CPU"),
    )
    assert sb.supports_n_cpu_moe("/bin/llama-server") is True

    monkeypatch.setattr(
        sb.subprocess, "run", lambda *a, **kw: _Result("  --n-cpu-moe N  keep experts")
    )
    assert sb.supports_n_cpu_moe("/bin/llama-server") is True

    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **kw: _Result("no such option here"))
    assert sb.supports_n_cpu_moe("/bin/llama-server") is False

    def _boom(*a, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(sb.subprocess, "run", _boom)
    assert sb.supports_n_cpu_moe("/bin/llama-server") is False


def test_server_command_maps_mlock_when_the_binary_supports_load_mode():
    command = sb.server_command(
        "/bin/llama-server", "/m.gguf", _plan(use_mlock=True), 1, load_mode_supported=True
    )

    assert command[command.index("-lm") + 1] == "mlock"


def test_server_command_skips_mlock_on_builds_without_load_mode():
    command = sb.server_command(
        "/bin/llama-server", "/m.gguf", _plan(use_mlock=True), 1, load_mode_supported=False
    )

    assert "-lm" not in command


def test_server_command_skips_mlock_when_the_plan_does_not_ask_for_it():
    command = sb.server_command(
        "/bin/llama-server", "/m.gguf", _plan(use_mlock=False), 1, load_mode_supported=True
    )

    assert "-lm" not in command


def test_supports_load_mode_reads_the_binary_help(monkeypatch):
    class _Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""

    monkeypatch.setattr(
        sb.subprocess,
        "run",
        lambda *a, **kw: _Result("  -lm, --load-mode MODE  model loading mode"),
    )
    assert sb._supports_load_mode("/bin/llama-server") is True

    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **kw: _Result("no such option here"))
    assert sb._supports_load_mode("/bin/llama-server") is False

    def _boom(*a, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(sb.subprocess, "run", _boom)
    assert sb._supports_load_mode("/bin/llama-server") is False


def test_server_command_leaves_f16_alone_and_can_disable_flash_attention():
    command = sb.server_command(
        "/bin/llama-server", "/m.gguf", _plan(kv_cache="f16", flash_attn=False), 1
    )

    assert "-ctk" not in command
    assert "-ctv" not in command
    assert command[command.index("-fa") + 1] == "off"


# --- HTTP surface ------------------------------------------------------------


def test_iter_sse_yields_chunks_and_stops_at_done():
    response = _Response(
        [
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n',
            b"\n",
            b"data: [DONE]\n",
            b'data: {"never":"read"}\n',
        ]
    )

    chunks = list(sb.ServerEngine._iter_sse(response))

    assert chunks == [{"choices": [{"delta": {"content": "a"}}]}]
    assert response.closed is True


def test_iter_sse_skips_unparsable_frames():
    response = _Response([b"data: not-json\n", b"data: [DONE]\n"])
    assert list(sb.ServerEngine._iter_sse(response)) == []


def test_tokenize_posts_text_and_returns_ints(monkeypatch):
    engine = _engine()
    seen = {}

    def fake_post(path, body, **kw):
        seen["path"], seen["body"] = path, body
        return _Payload({"tokens": [1, 2, "3"]})

    monkeypatch.setattr(engine, "_post", fake_post)

    assert engine.tokenize(b"hello") == [1, 2, 3]
    assert seen == {"path": "/tokenize", "body": {"content": "hello"}}


def test_chat_streams_and_forwards_usage_options(monkeypatch):
    engine = _engine()
    seen = {}

    def fake_post(path, body, **kw):
        seen.update(path=path, body=body)
        return _Response([b'data: {"choices":[]}\n', b"data: [DONE]\n"])

    monkeypatch.setattr(engine, "_post", fake_post)

    chunks = list(
        engine.chat(
            [{"role": "user", "content": "hi"}],
            stream=True,
            temperature=0.1,
            top_k=None,
            user="someone",  # not an OpenAI chat parameter llama-server knows
            stream_options={"include_usage": True},
        )
    )

    body = seen["body"]
    assert seen["path"] == "/v1/chat/completions"
    assert body["stream"] is True
    assert body["temperature"] == pytest.approx(0.1)
    assert body["stream_options"] == {"include_usage": True}
    assert "user" not in body
    assert "top_k" not in body  # None means "not set"
    # Documented defaults still apply.
    assert body["max_tokens"] == 2048
    assert chunks == [{"choices": []}]


def test_chat_without_stream_parses_the_json_body(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(
        engine,
        "_post",
        lambda path, body, **kw: _Payload({"choices": [{"message": {"content": "ok"}}]}),
    )

    out = engine.chat([{"role": "user", "content": "hi"}])

    assert out["choices"][0]["message"]["content"] == "ok"


# --- lifecycle ---------------------------------------------------------------


def test_close_terminates_the_child_process():
    engine = _engine()
    proc = _Proc()
    engine._proc = proc

    engine.close()

    assert proc.terminated is True
    assert engine._proc is None

    # Idempotent: a second close must not touch anything.
    engine.close()
    assert proc.terminated is True


def test_missing_binary_explains_what_to_install(monkeypatch):
    def missing(tool):
        raise OptimizerError("llama-server was not found.")

    monkeypatch.setattr(sb, "find_llama_tool", missing)

    with pytest.raises(RuntimeError, match="llama-server"):
        sb.ServerEngine(_plan(), "/m.gguf")


def test_gpu_start_failure_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(sb, "find_llama_tool", lambda tool: "/bin/llama-server")
    attempts = []

    def fake_start(self, plan):
        attempts.append(plan.gpu_layers)
        if plan.gpu_layers:
            raise RuntimeError("Failed to create llama_context")
        self.plan = plan

    monkeypatch.setattr(sb.ServerEngine, "_start", fake_start)

    engine = sb.ServerEngine(_plan(gpu_layers=7), "/m.gguf")

    # Walk the offload down (7, 6, 5, halved 3) before giving up the placement:
    # CPU-only measured 3.9 t/s where the last loadable split gave 5.0.
    assert attempts == [7, 6, 5, 2, 0]
    assert engine.plan.gpu_layers == 0


def test_gpu_start_failure_stops_at_the_last_split_that_starts(monkeypatch):
    monkeypatch.setattr(sb, "find_llama_tool", lambda tool: "/bin/llama-server")
    attempts = []

    def fake_start(self, plan):
        attempts.append(plan.gpu_layers)
        if plan.gpu_layers > 5:
            raise RuntimeError("Failed to create llama_context")
        self.plan = plan

    monkeypatch.setattr(sb.ServerEngine, "_start", fake_start)

    engine = sb.ServerEngine(_plan(gpu_layers=7, note="Hybrid: 7/40 layers"), "/m.gguf")

    assert attempts == [7, 6, 5]
    assert engine.plan.gpu_layers == 5


def test_placement_is_corrected_when_llama_server_ignores_gpu_layers(monkeypatch, capsys):
    engine = _engine()
    monkeypatch.setattr(
        engine,
        "_log_text",
        lambda limit=200_000: "warning: no usable GPU found, --gpu-layers option will be ignored\n",
    )

    plan = engine._adopt_placement(_plan(gpu_layers=7))

    assert plan.gpu_layers == 0
    assert plan.offload_kqv is False
    assert "CPU ONLY" in plan.note
    assert "ignored --gpu-layers 7" in capsys.readouterr().out


def test_placement_is_kept_when_the_child_uses_the_gpu(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(
        engine, "_log_text", lambda limit=200_000: "llm_load_tensors: offloaded 7/40 layers"
    )

    plan = engine._adopt_placement(_plan(gpu_layers=7))

    assert plan.gpu_layers == 7
    assert plan.offload_kqv is True


def test_server_command_keeps_one_context_per_slot():
    # -c is the whole window and llama.cpp splits it across slots, so N slots
    # need N windows to leave each request the context the planner chose.
    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1, parallel=4)

    assert command[command.index("-np") + 1] == "4"
    assert command[command.index("-c") + 1] == str(8192 * 4)


def test_server_command_always_states_the_slot_count():
    # These builds default to 4 slots, and a slot only receives 1/N of -c, so
    # an omitted -np would hand each request a quarter of the planned window.
    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1, parallel=1)

    assert command[command.index("-np") + 1] == "1"
    assert command[command.index("-c") + 1] == "8192"


def test_chat_sse_forwards_the_child_frames_unchanged(monkeypatch):
    engine = _engine()
    frames = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
        b"\n",
        b'data: {"choices":[],"usage":{"prompt_tokens":3}}\n',
        b"\n",
        b"data: [DONE]\n",
        b"\n",
        b"data: [DONE]\n",
    ]

    class _Raw:
        """What urllib hands back: read1 returns whatever has arrived."""

        closed = False

        def __init__(self, chunks):
            self.chunks = list(chunks)

        def read1(self, n):
            return self.chunks.pop(0) if self.chunks else b""

        def close(self):
            self.closed = True

    raw = _Raw([b"".join(frames[:3]), b"".join(frames[3:])])
    monkeypatch.setattr(engine, "_post", lambda path, body, **kw: raw)

    # Byte-for-byte, up to and including [DONE]; the child's trailing bytes are
    # never read.
    assert list(engine.chat_sse([{"role": "user", "content": "hi"}])) == [
        b"".join(frames[:3]),
        b"".join(frames[3:]),
    ]
    assert raw.closed is True


def test_server_command_takes_k_and_v_cache_types_separately():
    command = sb.server_command(
        "/bin/llama-server",
        "/m.gguf",
        _plan(kv_cache="q8_0", kv_cache_k="q8_0", kv_cache_v="q4_0"),
        1,
    )

    assert command[command.index("-ctk") + 1] == "q8_0"
    assert command[command.index("-ctv") + 1] == "q4_0"


def test_server_command_parks_kv_in_ram_only_with_layers_on_the_gpu():
    # offload_kqv=False with gpu_layers>0 is the KV-in-RAM axis
    # (`--no-kv-offload`). A CPU-only plan keeps the command line it had.
    parked = sb.server_command("/bin/llama-server", "/m.gguf", _plan(offload_kqv=False), 1)
    assert "--no-kv-offload" in parked

    cpu_only = sb.server_command(
        "/bin/llama-server", "/m.gguf", _plan(gpu_layers=0, offload_kqv=False), 1
    )
    assert "--no-kv-offload" not in cpu_only

    assert "--no-kv-offload" not in sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1)


def test_hybrid_split_disables_cuda_graphs():
    """A hybrid plan already spent the VRAM budget, so graphs can OOM a prompt.

    Measured on a 4 GB card: 18/40 layers loaded, served short prompts, then
    died in cudaGraphInstantiate on the first ~3k-token prompt.
    """
    env = sb.server_env(_plan(gpu_layers=7, n_layers=40), base={})
    assert env["GGML_CUDA_DISABLE_GRAPHS"] == "1"


def test_full_offload_and_cpu_only_keep_cuda_graphs():
    assert "GGML_CUDA_DISABLE_GRAPHS" not in sb.server_env(
        _plan(gpu_layers=40, n_layers=40), base={}
    )
    # A hand-built plan with no layer count is not a hybrid split either.
    assert "GGML_CUDA_DISABLE_GRAPHS" not in sb.server_env(_plan(gpu_layers=7), base={})
    assert "GGML_CUDA_DISABLE_GRAPHS" not in sb.server_env(
        _plan(gpu_layers=0, n_layers=40), base={}
    )


def test_moe_plan_keeps_cuda_graphs():
    """-ncmoe keeps every attention layer on the card: nothing sits at the edge."""
    plan = _plan(gpu_layers=3, n_layers=40, moe_cpu_layers=20)
    assert "GGML_CUDA_DISABLE_GRAPHS" not in sb.server_env(plan, base={})


def test_a_caller_graph_setting_wins():
    env = sb.server_env(_plan(gpu_layers=7, n_layers=40), base={"GGML_CUDA_DISABLE_GRAPHS": "0"})
    assert env["GGML_CUDA_DISABLE_GRAPHS"] == "0"


def test_start_passes_the_environment_to_the_child(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(sb, "_free_port", lambda: 4321)
    monkeypatch.setattr(sb.config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(sb, "server_command", lambda *a, **k: ["/bin/llama-server"])
    monkeypatch.setattr(sb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sb.ServerEngine, "_wait_ready", lambda self, proc, port: None)
    monkeypatch.setattr(sb.ServerEngine, "_adopt_placement", lambda self, plan: plan)

    engine = _engine()
    engine._binary = "/bin/llama-server"
    engine._model_path = "/m.gguf"
    engine._fit_supported = False
    engine._spec_type = "none"
    engine._spec_type_supported = False
    engine._moe_cache_supported = False
    engine.supports_embeddings = False
    engine.parallel = 1
    engine._start(_plan(gpu_layers=7, n_layers=40))

    assert captured["env"]["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    # The rest of the environment still reaches the child.
    assert "PATH" in captured["env"]


def test_supports_n_cpu_ffn_reads_the_binary_help(monkeypatch):
    class _Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""

    monkeypatch.setattr(
        sb.subprocess,
        "run",
        lambda *a, **kw: _Result("  -ncffn, --n-cpu-ffn N  keep dense FFN on CPU"),
    )
    assert sb.supports_n_cpu_ffn("/bin/llama-server") is True

    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **kw: _Result("  --n-cpu-ffn N  keep FFN"))
    assert sb.supports_n_cpu_ffn("/bin/llama-server") is True

    # -ncmoe is a different flag: a build with only that one cannot run a
    # tensor split, and passing it would kill the child.
    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **kw: _Result("  -ncmoe N"))
    assert sb.supports_n_cpu_ffn("/bin/llama-server") is False

    def _boom(*a, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(sb.subprocess, "run", _boom)
    assert sb.supports_n_cpu_ffn("/bin/llama-server") is False


def test_server_command_gives_a_tensor_split_the_full_layer_stack():
    """-ncffn is a tensor override: -ngl has to leave every layer to llama.cpp.

    Measured on the reference 4 GB card: -ngl 40 -ncffn 38 keeps all 40
    attention blocks and 2 FFN blocks on the card and decodes 4.89 t/s with a
    filled window, where the classic 17/40 split decodes 3.44 t/s.
    """
    plan = _plan(gpu_layers=17, n_layers=40, n_cpu_ffn=36)
    command = sb.server_command("/bin/llama-server", "/m.gguf", plan, 1)

    # llama.cpp counts the output layer past block_count: full stack is 41.
    assert command[command.index("-ngl") + 1] == "41"
    assert command[command.index("-ncffn") + 1] == "36"


def test_server_command_has_no_ncffn_without_a_tensor_split():
    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(gpu_layers=17), 1)
    assert "-ncffn" not in command
    assert command[command.index("-ngl") + 1] == "17"


def test_tensor_split_plan_gives_up_cuda_graphs():
    """Same reasoning as the classic hybrid: the budget went to placement.

    Measured at a filled window: 4.97 t/s with capture on against 4.89 t/s with
    it off, so the flag costs noise — and a plan that fills the budget cannot
    promise the 780 MiB of headroom that measurement had.
    """
    plan = _plan(gpu_layers=17, n_layers=40, n_cpu_ffn=36)
    assert sb.server_env(plan, base={})["GGML_CUDA_DISABLE_GRAPHS"] == "1"


def test_every_ffn_block_on_the_gpu_is_not_a_hybrid_split():
    # n_cpu_ffn=0 leaves llama.cpp nothing on the CPU side, so capture has a
    # whole card to work with; the graphs stay.
    plan = _plan(gpu_layers=41, n_layers=40, n_cpu_ffn=0)
    assert "GGML_CUDA_DISABLE_GRAPHS" not in sb.server_env(plan, base={})
