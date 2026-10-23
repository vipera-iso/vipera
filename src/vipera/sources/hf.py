"""Hugging Face Hub source.

`huggingface_hub` is imported inside the functions on purpose: it costs ~120 ms
to import and nothing else in the CLI needs it, so a `vipera --version` should
not pay for it.
"""

from __future__ import annotations

from pathlib import Path

from vipera.sources.protocol import FileInfo, SearchResult

NAME = "hf"
SCHEMES = ("hf", "huggingface")


def search(query: str, *, limit: int = 10) -> list[SearchResult]:
    from huggingface_hub import HfApi

    models = HfApi().list_models(search=query, filter="gguf", limit=limit)
    return [SearchResult(ref=m.id, title=m.id, downloads=m.downloads or 0) for m in models]


def resolve(ref: str) -> list[FileInfo]:
    """List the repo's .gguf files, with sizes when the API reports them."""
    from huggingface_hub import HfApi, list_repo_files

    files: list[FileInfo] = []
    try:
        for entry in HfApi().list_repo_tree(ref, recursive=True):
            name = getattr(entry, "path", None)
            if not name or not name.endswith(".gguf"):
                continue
            files.append(FileInfo(path=name, size_bytes=int(getattr(entry, "size", 0) or 0)))
    except Exception:
        # Older Hub clients, or a repo whose tree cannot be walked.
        files = [
            FileInfo(path=name) for name in list_repo_files(repo_id=ref) if name.endswith(".gguf")
        ]
    return files


def download(ref: str, file: FileInfo, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=ref, filename=file.path, local_dir=dest))
