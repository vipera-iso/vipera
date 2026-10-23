"""Where model files come from.

A source answers three questions: what matches a keyword, what files a
reference has, and how to fetch one. Only `hf` is wired today; the protocol is
what lets `ollama`, `s3` or `url` be added without touching the commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    ref: str
    title: str = ""
    downloads: int = 0
    formats: tuple[str, ...] = ("gguf",)
    size_gb: float | None = None


@dataclass(frozen=True)
class FileInfo:
    path: str
    size_bytes: int = 0
    format: str = "gguf"


class ModelSource(Protocol):
    name: str
    schemes: tuple[str, ...]

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]: ...

    def resolve(self, ref: str) -> list[FileInfo]: ...

    def download(self, ref: str, file: FileInfo, dest: Path) -> Path: ...
