"""The local Ollama store: import a model Ollama already pulled.

Ollama keeps every layer as a content-addressed blob under
``$OLLAMA_MODELS/blobs/`` and describes a model with a small JSON manifest.
An ``ollama:`` reference therefore needs no network: resolve reads the
manifest, and download links the blob into the models directory (falling back
to a copy when the two live on different filesystems).

Two layouts exist in the wild and both are read: current Ollama writes
``manifests-v2/<host>/<path>/<tag>`` with host ``ollama.com``; older releases
write ``manifests/registry.ollama.ai/<path>/<tag>``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from vipera.sources.protocol import FileInfo, SearchResult

NAME = "ollama"
SCHEMES = ("ollama",)

#: Manifest roots, newest layout first, and the hosts each may store under.
_MANIFEST_DIRS = ("manifests-v2", "manifests")
_HOSTS = ("ollama.com", "registry.ollama.ai")

_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


def models_dir() -> Path:
    """Where Ollama keeps its store; `OLLAMA_MODELS` overrides the default."""
    return Path(os.environ.get("OLLAMA_MODELS") or (Path.home() / ".ollama" / "models"))


def _split_ref(ref: str) -> tuple[str, str]:
    """Split ``name:tag``, ignoring a colon that belongs to a host:port."""
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    if colon > slash:
        name, _, tag = ref.rpartition(":")
        return name, tag or "latest"
    return ref, "latest"


def _manifest_paths(ref: str) -> list[Path]:
    """Every path the manifest for ``ref`` could have, newest layout first."""
    name, tag = _split_ref(ref)
    root = models_dir()
    first = name.split("/", 1)[0]

    candidates: list[Path] = []
    if "." in first and "/" in name:
        # A registry host is part of the name: hf.co/user/model:q4.
        host, path = name.split("/", 1)
        names = [path]
        hosts = [host]
    else:
        names = [name] if "/" in name else [f"library/{name}"]
        hosts = list(_HOSTS)

    for directory in _MANIFEST_DIRS:
        for host in hosts:
            for stored in names:
                candidates.append(root / directory / host / stored / tag)
    return candidates


def _read_manifest(ref: str) -> dict:
    for path in _manifest_paths(ref):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"unreadable Ollama manifest {path}: {exc}") from exc
        if isinstance(data, dict) and isinstance(data.get("layers"), list):
            return data
        raise ValueError(f"unexpected Ollama manifest shape in {path}")
    raise FileNotFoundError(
        f"'{ref}' is not in the local Ollama store ({models_dir()}). "
        f"Run `ollama pull {ref}` first, or check `ollama list`."
    )


def _model_layers(manifest: dict) -> list[dict]:
    layers = [
        layer
        for layer in manifest.get("layers") or []
        if str(layer.get("mediaType") or "").endswith(".model")
    ]
    if not layers:
        media = {str(layer.get("mediaType")) for layer in manifest.get("layers") or []}
        raise ValueError(
            "this Ollama manifest has no GGUF model layer "
            f"(layers: {', '.join(sorted(media)) or 'none'}); "
            "Vipera can only import GGUF models"
        )
    return layers


def _blob_path(digest: str) -> Path:
    # "sha256:abcd..." on disk is "sha256-abcd...".
    return models_dir() / "blobs" / str(digest).replace(":", "-", 1)


def _filename(layer: dict, ref: str) -> str:
    """Prefer the GGUF's original name (it carries the quant), else derive one."""
    source = layer.get("from")
    if isinstance(source, str) and source.endswith(".gguf"):
        return Path(source).name
    name, tag = _split_ref(ref)
    stem = name.rsplit("/", 1)[-1].replace("/", "-")
    return f"{stem}-{tag}.gguf" if tag != "latest" else f"{stem}.gguf"


def search(query: str, *, limit: int = 10) -> list[SearchResult]:
    """List models present in the local store whose name matches ``query``."""
    root = models_dir()
    needle = query.lower()
    results: dict[str, SearchResult] = {}

    for directory in _MANIFEST_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            parts = path.relative_to(base).parts
            if len(parts) < 4:
                continue
            host, *rest = parts
            tag = rest[-1]
            name = "/".join(rest[:-1])
            if name.startswith("library/"):
                name = name[len("library/") :]
            ref = f"{name}:{tag}"
            if needle not in name.lower() or ref in results:
                continue
            try:
                manifest = json.loads(path.read_text())
                layers = _model_layers(manifest)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            size = sum(int(layer.get("size") or 0) for layer in layers)
            results[ref] = SearchResult(ref=ref, title=f"{name} ({host})", size_gb=size / 1e9)
            if len(results) >= limit:
                return sorted(results.values(), key=lambda item: item.ref)
    return sorted(results.values(), key=lambda item: item.ref)


def resolve(ref: str) -> list[FileInfo]:
    """Describe the GGUF layer(s) of a model Ollama already has on disk."""
    manifest = _read_manifest(ref)
    files = []
    for layer in _model_layers(manifest):
        digest = layer.get("digest")
        if not isinstance(digest, str) or not _blob_path(digest).is_file():
            raise FileNotFoundError(
                f"'{ref}' is listed in its Ollama manifest but the model blob is "
                f"missing ({digest}); run `ollama pull {ref}` to restore it."
            )
        files.append(
            FileInfo(
                path=_filename(layer, ref),
                size_bytes=int(layer.get("size") or 0),
                format="gguf",
            )
        )
    return files


def download(ref: str, file: FileInfo, dest: Path) -> Path:
    """Bring the blob into ``dest`` without downloading or copying if possible."""
    manifest = _read_manifest(ref)
    layer = next(
        (item for item in _model_layers(manifest) if _filename(item, ref) == file.path),
        None,
    )
    if layer is None:
        raise FileNotFoundError(f"'{ref}' no longer has a layer named {file.path}")
    blob = _blob_path(str(layer.get("digest")))
    if not blob.is_file():
        raise FileNotFoundError(f"Ollama blob for '{ref}' is missing: {blob}")

    dest.mkdir(parents=True, exist_ok=True)
    target = dest / file.path
    if target.exists():
        if target.stat().st_size == blob.stat().st_size:
            return target
        target.unlink()
    try:
        # A hardlink costs no disk and stays valid if Ollama later removes the
        # model (the inode survives as long as this name does).
        os.link(blob, target)
    except OSError:
        shutil.copy2(blob, target)
    return target
