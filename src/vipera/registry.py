import json
import os
from datetime import UTC, datetime

from vipera.config import REGISTRY_FILE, ensure_dirs, quarantine_corrupt


def _load() -> dict:
    ensure_dirs()
    if REGISTRY_FILE.exists():
        try:
            data = json.loads(REGISTRY_FILE.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            quarantine_corrupt(REGISTRY_FILE, exc)
        else:
            if isinstance(data, dict):
                return data
            quarantine_corrupt(REGISTRY_FILE, ValueError("not a JSON object"))
    return {}


def _save(data: dict) -> None:
    ensure_dirs()
    tmp = REGISTRY_FILE.with_suffix(REGISTRY_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, REGISTRY_FILE)


def add_model(name: str, path, meta: dict | None = None) -> dict:
    data = _load()
    data[name] = {
        "name": name,
        "path": str(path),
        "added": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "meta": meta or {},
    }
    _save(data)
    return data[name]


def list_models() -> list[dict]:
    return list(_load().values())


def get_model(name: str) -> dict | None:
    return _load().get(name)


def remove_model(name: str) -> bool:
    data = _load()
    if name in data:
        del data[name]
        _save(data)
        return True
    return False
