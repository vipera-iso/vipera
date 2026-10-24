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

    # A MoE plan keeps all 24 layers for dense tensors and moves the first 12
    # layers' routed experts to the CPU.
    assert command[command.index("-ngl") + 1] == "24"
    assert command[command.index("-ncmoe") + 1] == "12"
    # -ub is the prefill lever and -b must be at least as large.
    assert command[command.index("-ub") + 1] == "896"
    assert command[command.index("-b") + 1] == "896"


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

    def fake_post(path, body):
        seen["path"], seen["body"] = path, body
        return _Payload({"tokens": [1, 2, "3"]})

    monkeypatch.setattr(engine, "_post", fake_post)

    assert engine.tokenize(b"hello") == [1, 2, 3]
    assert seen == {"path": "/tokenize", "body": {"content": "hello"}}


def test_chat_streams_and_forwards_usage_options(monkeypatch):
    engine = _engine()
    seen = {}

    def fake_post(path, body):
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
        lambda path, body: _Payload({"choices": [{"message": {"content": "ok"}}]}),
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


def test_server_command_omits_np_for_a_single_slot():
    command = sb.server_command("/bin/llama-server", "/m.gguf", _plan(), 1, parallel=1)

    assert "-np" not in command
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
    monkeypatch.setattr(engine, "_post", lambda path, body: raw)

    # Byte-for-byte, up to and including [DONE]; the child's trailing bytes are
    # never read.
    assert list(engine.chat_sse([{"role": "user", "content": "hi"}])) == [
        b"".join(frames[:3]),
        b"".join(frames[3:]),
    ]
    assert raw.closed is True
