"""A direct download URL, e.g. a release asset or an internal mirror."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from vipera.sources.protocol import FileInfo, SearchResult

NAME = "url"
SCHEMES = ("http", "https")


def search(query: str, *, limit: int = 10) -> list[SearchResult]:
    # A URL is not searchable; the scheme router resolves it directly.
    return []


def resolve(ref: str) -> list[FileInfo]:
    request = urllib.request.Request(ref, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers.get("Content-Length") or 0)
    name = Path(urlparse(ref).path).name or "download.gguf"
    return [FileInfo(path=name, size_bytes=size, format=Path(name).suffix.lstrip("."))]


def download(ref: str, file: FileInfo, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / file.path
    with urllib.request.urlopen(ref, timeout=60) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    return target
