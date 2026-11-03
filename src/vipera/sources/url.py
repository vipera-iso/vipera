"""A direct download URL, e.g. a release asset or an internal mirror.

The reference is the one input in this package that names an arbitrary host, so
both entry points treat it as hostile:

* the target must be http(s), and must not resolve to a private, loopback,
  link-local or reserved address. A model URL fetches public bytes; pointing it
  at ``169.254.169.254`` or at a service on ``127.0.0.1`` is what a bug or a
  hostile config would do, not what a user means;
* a redirect is checked before it is followed, hop by hop, so a public mirror
  cannot forward the CLI to an internal service. The stdlib handler already
  refuses a hop to a scheme other than http(s)/ftp and caps the chain at ten;
* the body is written under a ceiling, so a server that answers ``200`` and
  never stops cannot fill the disk.

What this does not defend against is DNS rebinding between the check and the
connect: pinning the address would mean dialling the IP with a ``Host`` header,
which is more machinery than a CLI download earns. The window is written down
here rather than pretended away.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import shutil
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from vipera.sources.protocol import FileInfo, SearchResult

NAME = "url"
SCHEMES = ("http", "https")

#: A GGUF larger than this is not a model someone serves on a desktop; it is a
#: server that did not stop sending. 128 GiB clears the biggest files in
#: circulation (a 120B Q2 is ~62 GB) with room to spare.
MAX_DOWNLOAD_BYTES = 128 * 1024**3
#: Disk left untouched, so a body that does not fit fails with our message
#: rather than with ENOSPC halfway through 60 GB.
_DISK_MARGIN_BYTES = 2 * 1024**3
_CHUNK_BYTES = 1024 * 1024


class UnsafeURLError(OSError):
    """The reference is not something this gateway will fetch.

    An ``OSError`` because that is what a failed download already raises: the
    callers treat "cannot fetch this" as one failure, not two.
    """


def _public_host(host: str) -> None:
    """Refuse a host that resolves to anything but a public address.

    Every address the name resolves to is checked, not just the first: a name
    with one public A record and one private AAAA record would otherwise pass
    on the record that happens to sort first.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"{host}: {exc}") from exc
    if not infos:
        raise UnsafeURLError(f"{host} does not resolve")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # is_global is False for private, loopback, link-local, CGNAT, reserved,
        # unspecified and multicast alike, which is the whole list we want.
        if not ip.is_global:
            raise UnsafeURLError(f"{host} resolves to {ip}, which is not a public address")


def _checked(ref: str) -> None:
    """Raise unless `ref` is an http(s) URL for a public host."""
    parsed = urlparse(ref)
    if parsed.scheme not in SCHEMES:
        allowed = "/".join(SCHEMES)
        raise UnsafeURLError(f"{ref}: only {allowed} references can be downloaded")
    if not parsed.hostname:
        raise UnsafeURLError(f"{ref}: no host to fetch")
    _public_host(parsed.hostname)


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to a target that passes the same check.

    ``super()`` keeps the stdlib's own limits (ten hops, http(s)/ftp only); this
    adds the address check, so "the mirror moved" cannot become "fetch from the
    machine's own network".
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _checked(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_CheckedRedirect)


def _limit_for(dest: Path, declared: int) -> int:
    """Bytes we are willing to write: the ceiling, or the room actually left.

    A declared size over the bound is refused before a byte is written. An
    undeclared one is still bounded, because the counter in ``download`` stops
    the write at the same number.
    """
    limit = MAX_DOWNLOAD_BYTES
    with contextlib.suppress(OSError):
        free = shutil.disk_usage(dest).free
        limit = min(limit, max(0, free - _DISK_MARGIN_BYTES))
    if declared > limit:
        raise UnsafeURLError(
            f"a {declared}-byte download does not fit: at most {limit} bytes may be written"
        )
    return limit


def search(query: str, *, limit: int = 10) -> list[SearchResult]:
    # A URL is not searchable; the scheme router resolves it directly.
    return []


def resolve(ref: str) -> list[FileInfo]:
    _checked(ref)
    request = urllib.request.Request(ref, method="HEAD")
    with _OPENER.open(request, timeout=30) as response:
        size = int(response.headers.get("Content-Length") or 0)
    if size > MAX_DOWNLOAD_BYTES:
        raise UnsafeURLError(f"{ref}: {size} bytes is over the {MAX_DOWNLOAD_BYTES}-byte ceiling")
    name = Path(urlparse(ref).path).name or "download.gguf"
    return [FileInfo(path=name, size_bytes=size, format=Path(name).suffix.lstrip("."))]


def download(ref: str, file: FileInfo, dest: Path) -> Path:
    """Fetch `ref` into `dest/file.path`, atomically, checked and size-capped.

    Written to a per-process `.part` beside the target and renamed only when the
    whole body arrived, so a dropped connection or a Ctrl-C cannot leave a truncated file
    at the path the registry treats as a model: `model add` only warns when a
    GGUF will not parse, and a half file is exactly what produces that warning
    and then a failed load days later. The size check is the same guard for a
    server that answered 200 with a short body; `FileInfo.size_bytes` is 0 when
    the HEAD carried no Content-Length, and then there is nothing to compare.
    """
    _checked(ref)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / file.path
    # Unique per process: two `model add` runs fetching the same file used to
    # share one `.part`, and the loser could rename a half-written body over the
    # winner's target. It stays in `dest`, so the rename below is still atomic.
    partial = target.with_name(f"{target.name}.{os.getpid()}.part")
    limit = _limit_for(dest, file.size_bytes)
    written = 0
    try:
        with _OPENER.open(ref, timeout=60) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                # Checked per chunk rather than against Content-Length: the
                # header can lie, and a chunked body has none at all.
                if written > limit:
                    raise UnsafeURLError(
                        f"{ref}: the body passed the {limit}-byte limit, so the download stopped"
                    )
                out.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    written = partial.stat().st_size
    if file.size_bytes and written != file.size_bytes:
        partial.unlink(missing_ok=True)
        raise OSError(f"downloaded {written} bytes, expected {file.size_bytes}")
    partial.replace(target)
    return target
