"""A direct download URL, e.g. a release asset or an internal mirror."""

from __future__ import annotations

import os
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
    """Fetch `ref` into `dest/file.path`, atomically and size-checked.

    Written to a per-process `.part` beside the target and renamed only when the
    whole body arrived, so a dropped connection or a Ctrl-C cannot leave a truncated file
    at the path the registry treats as a model: `model add` only warns when a
    GGUF will not parse, and a half file is exactly what produces that warning
    and then a failed load days later. The size check is the same guard for a
    server that answered 200 with a short body; `FileInfo.size_bytes` is 0 when
    the HEAD carried no Content-Length, and then there is nothing to compare.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / file.path
    # Unique per process: two `model add` runs fetching the same file used to
    # share one `.part`, and the loser could rename a half-written body over the
    # winner's target. It stays in `dest`, so the rename below is still atomic.
    partial = target.with_name(f"{target.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(ref, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    written = partial.stat().st_size
    if file.size_bytes and written != file.size_bytes:
        partial.unlink(missing_ok=True)
        raise OSError(f"downloaded {written} bytes, expected {file.size_bytes}")
    partial.replace(target)
    return target
