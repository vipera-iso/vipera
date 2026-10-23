"""The installed-model registry.

On disk the file is versioned (``_schema``) so a future change can be migrated
instead of guessed at. 1.x wrote a flat ``{alias: entry}`` map with no version;
that shape is still read and is upgraded on the next write, so an old install
keeps every model without a migration step.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from vipera.config import REGISTRY_FILE, ensure_dirs, quarantine_corrupt

SCHEMA_VERSION = 2

# Fields a 1.x entry has, in the order a migrated file should keep them.
_V1_FIELDS = ("name", "path", "added", "meta")


@dataclass(frozen=True)
class ModelSpec:
    """One installed model, as stored and as callers see it."""

    alias: str
    path: Path
    # Where the file came from. "local" is what a 1.x entry becomes.
    source: str = "local"
    source_ref: str | None = None
    format: str = "gguf"
    meta: dict = field(default_factory=dict)
    added: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_raw() -> dict:
    ensure_dirs()
    if not REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(REGISTRY_FILE.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        quarantine_corrupt(REGISTRY_FILE, exc)
        return {}
    if not isinstance(data, dict):
        quarantine_corrupt(REGISTRY_FILE, ValueError("not a JSON object"))
        return {}
    return data


def _spec_from_entry(alias: str, entry: dict) -> ModelSpec | None:
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        return None
    return ModelSpec(
        alias=str(entry.get("alias") or entry.get("name") or alias),
        path=Path(path).expanduser(),
        source=str(entry.get("source") or "local"),
        source_ref=entry.get("source_ref"),
        format=str(entry.get("format") or "gguf"),
        meta=dict(entry.get("meta") or {}),
        added=str(entry.get("added") or ""),
    )


def _load() -> dict[str, ModelSpec]:
    """Read every entry, accepting both the v2 and the 1.x file shape."""
    raw = _read_raw()
    entries = raw.get("models")
    if not isinstance(entries, dict):
        # 1.x: a flat alias -> entry map, no version key.
        entries = {k: v for k, v in raw.items() if isinstance(v, dict)}

    specs: dict[str, ModelSpec] = {}
    for alias, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        spec = _spec_from_entry(alias, entry)
        if spec is not None:
            specs[alias] = spec
    return specs


def _as_entry(spec: ModelSpec) -> dict:
    entry = {
        "alias": spec.alias,
        "path": str(spec.path),
        "source": spec.source,
        "format": spec.format,
        "added": spec.added,
        "meta": spec.meta,
    }
    if spec.source_ref:
        entry["source_ref"] = spec.source_ref
    return entry


def _save(specs: dict[str, ModelSpec]) -> None:
    ensure_dirs()
    payload = {"_schema": SCHEMA_VERSION, "models": {k: _as_entry(v) for k, v in specs.items()}}
    # Write-then-rename: a crash mid-write must not truncate the registry.
    tmp = REGISTRY_FILE.with_suffix(REGISTRY_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, REGISTRY_FILE)


def add_model(
    name: str,
    path,
    meta: dict | None = None,
    *,
    source: str = "local",
    source_ref: str | None = None,
    format: str = "gguf",
) -> ModelSpec:
    specs = _load()
    spec = ModelSpec(
        alias=name,
        path=Path(path).expanduser(),
        source=source,
        source_ref=source_ref,
        format=format,
        meta=meta or {},
        added=_now(),
    )
    specs[name] = spec
    _save(specs)
    return spec


def list_models() -> list[ModelSpec]:
    return list(_load().values())


def get_model(name: str) -> ModelSpec | None:
    return _load().get(name)


def remove_model(name: str) -> bool:
    specs = _load()
    if name not in specs:
        return False
    del specs[name]
    _save(specs)
    return True
