import hashlib
import json
import os

import pytest

from vipera.sources import for_ref, hf, local, ollama, url

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
