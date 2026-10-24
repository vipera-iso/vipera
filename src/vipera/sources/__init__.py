"""Built-in model sources. Each module implements the `ModelSource` protocol.

`for_ref` is the router: a reference is either a plain path (no source, the
command handles it) or carries a scheme, so `vipera model import` does not
have to know which source understands the string.
"""

from vipera.sources import hf, local, ollama, url
from vipera.sources.protocol import FileInfo, ModelSource, SearchResult

BUILTIN = {module.NAME: module for module in (hf, local, ollama, url)}

_BY_SCHEME = {scheme: module for module in BUILTIN.values() for scheme in module.SCHEMES}


def for_ref(ref: str):
    """Return ``(source module, ref without prefix)`` for a scheme reference.

    ``None`` as the source means "a plain path"; the caller keeps the original
    string. A full URL keeps its scheme, because the source downloads from it.
    """
    if "://" in ref:
        scheme = ref.split("://", 1)[0].lower()
        module = _BY_SCHEME.get(scheme)
        return (module, ref) if module is not None else (None, ref)
    head, sep, rest = ref.partition(":")
    if not sep:
        return None, ref
    module = _BY_SCHEME.get(head.lower())
    if module is None:
        return None, ref
    return module, rest


__all__ = [
    "BUILTIN",
    "FileInfo",
    "ModelSource",
    "SearchResult",
    "for_ref",
    "hf",
    "local",
    "ollama",
    "url",
]
