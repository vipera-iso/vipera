import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

VIPERA_HOME = Path(os.environ.get("VIPERA_HOME") or (Path.home() / ".vipera"))
MODELS_DIR = VIPERA_HOME / "models"
LOGS_DIR = VIPERA_HOME / "logs"
REGISTRY_FILE = VIPERA_HOME / "registry.json"
CONFIG_FILE = VIPERA_HOME / "config.json"

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8686,
    "default_model": None,
    "profile": "auto",
    "ctx": 8192,
    "kv_cache": "q8_0",
    "flash_attn": True,
    "n_batch": 512,
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
}

_KV_CACHE_TYPES = {"f32", "f16", "q4_0", "q4_1", "q8_0"}

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off", ""}

_TYPE_MAP = {
    "port": int,
    "ctx": int,
    "n_batch": int,
    "max_tokens": int,
    "temperature": float,
    "top_p": float,
    "repeat_penalty": float,
    "flash_attn": bool,
}


def ensure_dirs() -> None:
    for d in (VIPERA_HOME, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _coerce(cfg: dict) -> dict:
    for k, cast in _TYPE_MAP.items():
        if k in cfg and cfg[k] is not None:
            try:
                if cast is bool:
                    cfg[k] = str(cfg[k]).lower() in ("1", "true", "yes", "on")
                else:
                    cfg[k] = cast(cfg[k])
            except (ValueError, TypeError):
                pass
    if cfg.get("kv_cache") not in _KV_CACHE_TYPES:
        cfg["kv_cache"] = DEFAULT_CONFIG["kv_cache"]
    return cfg


def _parse_bool(value) -> bool:
    """Strictly parse a boolean, rejecting anything that is not unambiguous."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUE_STRINGS:
        return True
    if s in _FALSE_STRINGS:
        return False
    raise ValueError(f"expected a boolean, got '{value}'")


def _coerce_strict(cfg: dict) -> dict:
    """Raise ValueError on any invalid typed value."""
    for k, cast in _TYPE_MAP.items():
        if k in cfg and cfg[k] is not None:
            try:
                if cast is bool:
                    cfg[k] = _parse_bool(cfg[k])
                else:
                    cfg[k] = cast(cfg[k])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{k}: invalid {cast.__name__} value '{cfg[k]}'") from exc
    if "kv_cache" in cfg and cfg["kv_cache"] not in _KV_CACHE_TYPES:
        valid = ", ".join(sorted(_KV_CACHE_TYPES))
        raise ValueError(f"kv_cache: must be one of {valid}")
    return cfg


def quarantine_corrupt(path: Path, exc: Exception) -> Path:
    """Keep an unreadable JSON file before anything overwrites it.

    Loads fall back to defaults, so without this copy the next
    ``system config``/``model import`` would silently erase the user's file.
    """
    backup = path.with_name(path.name + ".corrupt")
    with contextlib.suppress(OSError):
        shutil.copy2(path, backup)
    print(
        f"vipera: {path} is unreadable ({exc}); using defaults. Original kept at {backup}",
        file=sys.stderr,
    )
    return backup


def load_config() -> dict:
    ensure_dirs()
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            quarantine_corrupt(CONFIG_FILE, exc)
        else:
            if isinstance(data, dict):
                return _coerce({**DEFAULT_CONFIG, **data})
            quarantine_corrupt(CONFIG_FILE, ValueError("not a JSON object"))
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    ensure_dirs()
    # Write-then-rename so a crash mid-write cannot leave a truncated config
    # that load_config() would silently replace with defaults.
    tmp = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, CONFIG_FILE)
