"""Built-in model sources. Each module implements the `ModelSource` protocol."""

from vipera.sources import hf, local, url
from vipera.sources.protocol import FileInfo, ModelSource, SearchResult

BUILTIN = {module.NAME: module for module in (hf, local, url)}

__all__ = ["BUILTIN", "FileInfo", "ModelSource", "SearchResult", "hf", "local", "url"]
