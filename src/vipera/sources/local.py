"""Local files: a path the user already has on disk."""

from __future__ import annotations

import shutil
from pathlib import Path

from vipera.config import MODELS_DIR
from vipera.sources.protocol import FileInfo, SearchResult

NAME = "local"
SCHEMES = ("local", "file")


def _file_info(path: Path) -> FileInfo:
    return FileInfo(path=str(path), size_bytes=path.stat().st_size, format=path.suffix.lstrip("."))


def search(query: str, *, limit: int = 10) -> list[SearchResult]:
    """Match installed files by name. Nothing leaves the machine."""
    if not MODELS_DIR.exists():
        return []
    needle = query.lower()
    matches = [
        p
        for p in sorted(MODELS_DIR.iterdir())
        if p.is_file() and p.suffix == ".gguf" and needle in p.name.lower()
    ]
    return [
        SearchResult(ref=str(p), title=p.name, size_gb=p.stat().st_size / 1e9)
        for p in matches[:limit]
    ]


def resolve(ref: str) -> list[FileInfo]:
    path = Path(ref).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {ref}")
    return [_file_info(path)]


def download(ref: str, file: FileInfo, dest: Path) -> Path:
    """Copy (or leave in place when it already lives in the models directory)."""
    source = Path(file.path).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / source.name
    if source.resolve() == target.resolve():
        return source
    shutil.copy2(source, target)
    return target
