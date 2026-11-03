import hashlib
import io
import json
import os
import socket

import pytest

from vipera.sources import for_ref, hf, local, ollama, url
from vipera.sources.protocol import FileInfo

GGUF_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def _blob(root, data: bytes) -> dict:
    """Write a layer blob and return the manifest layer entry for it."""
    digest = hashlib.sha256(data).hexdigest()
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "blobs" / f"sha256-{digest}").write_bytes(data)
    return {
        "mediaType": "application/vnd.ollama.image.model",
        "digest": f"sha256:{digest}",
        "size": len(data),
        "from": GGUF_NAME,
    }


def _manifest(root, relative: str, layers: list[dict]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "ollama-models"
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    return root


@pytest.fixture
def pulled(store):
    """A model pulled by current Ollama: manifests-v2/ollama.com/library."""
    layer = _blob(store, b"GGUF-payload")
    _manifest(
        store,
        "manifests-v2/ollama.com/library/qwen05/latest",
        [layer, {"mediaType": "application/vnd.ollama.image.params", "size": 18}],
    )
    return layer


# -- scheme routing ----------------------------------------------------------


def test_a_plain_path_is_not_a_source_reference():
    assert for_ref("/tmp/model.gguf") == (None, "/tmp/model.gguf")
    assert for_ref("./relative.gguf") == (None, "./relative.gguf")


def test_hf_prefix_selects_the_hugging_face_source():
    source, ref = for_ref("hf:Qwen/Qwen2.5-7B-Instruct-GGUF")

    assert source is hf
    assert ref == "Qwen/Qwen2.5-7B-Instruct-GGUF"


def test_local_prefix_selects_the_local_source():
    source, ref = for_ref("local:~/Downloads/model.gguf")

    assert source is local
    assert ref == "~/Downloads/model.gguf"


def test_ollama_prefix_keeps_the_tag_in_the_reference():
    # The scheme separator comes off, the tag separator stays.
    source, ref = for_ref("ollama:qwen2.5:7b")

    assert source is ollama
    assert ref == "qwen2.5:7b"


def test_a_url_keeps_its_scheme_because_the_source_downloads_from_it():
    ref = "https://example.com/model.gguf"

    source, resolved = for_ref(ref)

    assert source is url
    assert resolved == ref


def test_an_unknown_scheme_is_left_alone():
    assert for_ref("s3://bucket/model.gguf") == (None, "s3://bucket/model.gguf")
    assert for_ref("weird:thing") == (None, "weird:thing")


# -- Ollama store ------------------------------------------------------------


def test_ollama_resolve_reads_the_current_manifest_layout(store, pulled):
    files = ollama.resolve("qwen05")

    assert [f.path for f in files] == [GGUF_NAME]  # the GGUF's original name
    assert files[0].size_bytes == len(b"GGUF-payload")
    assert files[0].format == "gguf"


def test_ollama_resolve_reads_the_legacy_manifest_layout(store):
    layer = _blob(store, b"old-layout")
    _manifest(store, "manifests/registry.ollama.ai/library/tiny/latest", [layer])

    files = ollama.resolve("tiny:latest")

    assert [f.path for f in files] == [GGUF_NAME]


def test_ollama_resolve_derives_a_name_when_the_layer_has_none(store):
    layer = _blob(store, b"no-from")
    layer.pop("from")
    _manifest(store, "manifests-v2/ollama.com/library/qwen05/7b", [layer])

    assert [f.path for f in ollama.resolve("qwen05:7b")] == ["qwen05-7b.gguf"]


def test_ollama_resolve_supports_a_registry_host_in_the_name(store):
    layer = _blob(store, b"hf-model")
    _manifest(store, "manifests-v2/hf.co/user/repo/q4", [layer])

    assert [f.path for f in ollama.resolve("hf.co/user/repo:q4")] == [GGUF_NAME]


def test_ollama_resolve_explains_how_to_get_the_model(store):
    with pytest.raises(FileNotFoundError) as exc:
        ollama.resolve("qwen2.5:7b")

    assert "ollama pull qwen2.5:7b" in str(exc.value)


def test_ollama_resolve_rejects_a_manifest_without_a_model_layer(store):
    _manifest(
        store,
        "manifests-v2/ollama.com/library/embed/latest",
        [{"mediaType": "application/vnd.ollama.image.params", "size": 18}],
    )

    with pytest.raises(ValueError, match="no GGUF model layer"):
        ollama.resolve("embed")


def test_ollama_download_hardlinks_the_blob(store, pulled, tmp_path):
    dest = tmp_path / "models"
    blob = store / "blobs" / pulled["digest"].replace(":", "-")

    path = ollama.download("qwen05", ollama.resolve("qwen05")[0], dest)

    assert path.read_bytes() == b"GGUF-payload"
    # Same inode: nothing was copied and no second copy of the model exists.
    assert path.stat().st_ino == blob.stat().st_ino


def test_ollama_download_copies_when_linking_is_impossible(store, pulled, tmp_path, monkeypatch):
    def _no_link(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", _no_link)
    dest = tmp_path / "models"

    path = ollama.download("qwen05", ollama.resolve("qwen05")[0], dest)

    assert path.read_bytes() == b"GGUF-payload"
    assert (
        path.stat().st_ino != (store / "blobs" / pulled["digest"].replace(":", "-")).stat().st_ino
    )


def test_ollama_download_reuses_an_already_imported_file(store, pulled, tmp_path):
    dest = tmp_path / "models"
    dest.mkdir()
    existing = dest / GGUF_NAME
    existing.write_bytes(b"GGUF-payload")
    inode = existing.stat().st_ino

    path = ollama.download("qwen05", ollama.resolve("qwen05")[0], dest)

    assert path.stat().st_ino == inode, "an existing file must not be rewritten"


def test_ollama_search_lists_local_models(store):
    qwen = _blob(store, b"qwen")
    _manifest(store, "manifests-v2/ollama.com/library/qwen05/latest", [qwen])
    other = _blob(store, b"lfm")
    _manifest(store, "manifests-v2/ollama.com/library/lfm25/latest", [other])

    all_refs = [r.ref for r in ollama.search("")]
    assert all_refs == ["lfm25:latest", "qwen05:latest"]

    only_qwen = ollama.search("qwen")
    assert [r.ref for r in only_qwen] == ["qwen05:latest"]
    assert only_qwen[0].size_gb == pytest.approx(len(b"qwen") / 1e9)


# -- url downloads -----------------------------------------------------------


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host to a public address, so no test touches real DNS."""

    def _fake(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(url.socket, "getaddrinfo", _fake)


def _resolves_to(monkeypatch, address: str) -> None:
    """Make the next lookup answer `address` (v4 or v6)."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET

    def _fake(host, *args, **kwargs):
        return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 0, 0, 0))]

    monkeypatch.setattr(url.socket, "getaddrinfo", _fake)


class _Opener:
    """Stand-in for the module opener: every open() answers with `body`."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.seen: list = []

    def open(self, request, **_kwargs):
        self.seen.append(request)
        if isinstance(self.body, Exception):
            raise self.body
        return self.body if hasattr(self.body, "read") else io.BytesIO(self.body)


def _serve(monkeypatch, body: bytes) -> _Opener:
    opener = _Opener(body)
    monkeypatch.setattr(url, "_OPENER", opener)
    return opener


def test_url_download_writes_the_final_name_only_after_the_whole_body(
    monkeypatch, tmp_path, public_dns
):
    _serve(monkeypatch, b"GGUF" + b"\0" * 60)
    dest = tmp_path / "models"
    info = FileInfo(path="m.gguf", size_bytes=64)

    path = url.download("https://example.invalid/m.gguf", info, dest)

    assert path == dest / "m.gguf"
    assert path.read_bytes()[:4] == b"GGUF"
    assert not list(dest.glob("*.part"))


def test_url_download_rejects_a_short_body(monkeypatch, tmp_path, public_dns):
    # A server that answers 200 with a partial body used to leave the file at
    # the final name, where `model add` only warns and `serve start` fails.
    _serve(monkeypatch, b"GGUF" + b"\0" * 4)
    dest = tmp_path / "models"
    info = FileInfo(path="m.gguf", size_bytes=64)

    with pytest.raises(OSError):
        url.download("https://example.invalid/m.gguf", info, dest)

    assert not (dest / "m.gguf").exists()
    assert not list(dest.glob("*.part"))


def test_url_download_leaves_nothing_behind_when_the_connection_drops(
    monkeypatch, tmp_path, public_dns
):
    class _Drops(io.BytesIO):
        def read(self, *args):
            raise OSError("connection reset by peer")

    monkeypatch.setattr(url, "_OPENER", _Opener(_Drops(b"GGUF")))
    dest = tmp_path / "models"
    info = FileInfo(path="m.gguf", size_bytes=64)

    with pytest.raises(OSError):
        url.download("https://example.invalid/m.gguf", info, dest)

    assert not (dest / "m.gguf").exists()
    assert not list(dest.glob("*.part"))


def test_url_download_accepts_whatever_arrived_when_the_head_had_no_length(
    monkeypatch, tmp_path, public_dns
):
    # A HEAD with no Content-Length leaves size_bytes at 0, and then there is
    # nothing to compare: the body must be taken as it came rather than guessed
    # at, or every mirror that omits the header would fail to install.
    _serve(monkeypatch, b"GGUF")
    dest = tmp_path / "models"
    info = FileInfo(path="m.gguf")

    path = url.download("https://example.invalid/m.gguf", info, dest)

    assert path.read_bytes() == b"GGUF"
    assert not list(dest.glob("*.part"))


# -- url hardening: what the reference may point at -------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # the gateway's own port
        "10.1.2.3",  # a private network
        "169.254.169.254",  # cloud metadata
        "::1",  # loopback, v6
        "fd00::1",  # unique-local, v6
        "0.0.0.0",  # unspecified
    ],
)
def test_a_reference_that_resolves_inside_the_network_is_refused(monkeypatch, tmp_path, address):
    _resolves_to(monkeypatch, address)
    _serve(monkeypatch, b"GGUF")
    dest = tmp_path / "models"

    with pytest.raises(url.UnsafeURLError) as err:
        url.download("https://mirror.example/m.gguf", FileInfo(path="m.gguf"), dest)

    assert "not a public address" in str(err.value)
    assert not dest.exists() or not any(dest.iterdir())


def test_resolve_refuses_the_same_targets(monkeypatch):
    # The HEAD is a request too: validating only the download would still let a
    # caller probe an internal service through `model search <url>`.
    _resolves_to(monkeypatch, "127.0.0.1")
    _serve(monkeypatch, b"")

    with pytest.raises(url.UnsafeURLError):
        url.resolve("https://mirror.example/m.gguf")


def test_a_non_http_scheme_is_refused(monkeypatch):
    with pytest.raises(url.UnsafeURLError) as err:
        url.resolve("file:///etc/passwd")

    assert "only http/https" in str(err.value)


def test_a_redirect_to_a_private_host_is_refused(monkeypatch):
    # A public mirror that answers 302 towards 127.0.0.1 is the classic way past
    # a check that only looked at the URL the user typed.
    _resolves_to(monkeypatch, "127.0.0.1")
    request = url.urllib.request.Request("https://mirror.example/m.gguf")

    with pytest.raises(url.UnsafeURLError):
        url._CheckedRedirect().redirect_request(
            request, None, 302, "Found", {}, "http://127.0.0.1:8686/health"
        )


def test_a_redirect_to_a_public_host_is_followed(public_dns):
    request = url.urllib.request.Request("https://mirror.example/m.gguf")

    followed = url._CheckedRedirect().redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example/m.gguf"
    )

    assert followed.full_url == "https://cdn.example/m.gguf"


# -- url hardening: how much a body may write ------------------------------


def test_a_body_over_the_ceiling_is_stopped_mid_write(monkeypatch, tmp_path, public_dns):
    monkeypatch.setattr(url, "MAX_DOWNLOAD_BYTES", 10)
    _serve(monkeypatch, b"x" * 64)
    dest = tmp_path / "models"

    with pytest.raises(url.UnsafeURLError) as err:
        url.download("https://mirror.example/m.gguf", FileInfo(path="m.gguf"), dest)

    assert "byte limit" in str(err.value)
    assert not (dest / "m.gguf").exists()
    assert not list(dest.glob("*.part"))


def test_a_declared_size_over_the_ceiling_is_refused_before_writing(
    monkeypatch, tmp_path, public_dns
):
    monkeypatch.setattr(url, "MAX_DOWNLOAD_BYTES", 10)
    opener = _serve(monkeypatch, b"x" * 4)
    dest = tmp_path / "models"

    with pytest.raises(url.UnsafeURLError) as err:
        url.download("https://mirror.example/m.gguf", FileInfo(path="m.gguf", size_bytes=64), dest)

    assert "does not fit" in str(err.value)
    assert opener.seen == []  # nothing was fetched at all


def test_resolve_refuses_a_declared_size_over_the_ceiling(monkeypatch, public_dns):
    monkeypatch.setattr(url, "MAX_DOWNLOAD_BYTES", 10)

    class _Head(io.BytesIO):
        def __init__(self, body=b""):
            super().__init__(body)
            self.headers = {"Content-Length": "64"}

    monkeypatch.setattr(url, "_OPENER", _Opener(b""))
    monkeypatch.setattr(url._OPENER, "open", lambda *_a, **_k: _Head(b""))

    with pytest.raises(url.UnsafeURLError) as err:
        url.resolve("https://mirror.example/m.gguf")

    assert "ceiling" in str(err.value)
