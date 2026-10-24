import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

from vipera.backends.names import is_known as _is_known_backend
from vipera.backends.names import names as backend_names
from vipera.planner import profile_names
from vipera.protocols.names import is_known as _is_known_protocol
from vipera.protocols.names import names as protocol_names

VIPERA_HOME = Path(os.environ.get("VIPERA_HOME") or (Path.home() / ".vipera"))
MODELS_DIR = VIPERA_HOME / "models"
LOGS_DIR = VIPERA_HOME / "logs"
REGISTRY_FILE = VIPERA_HOME / "registry.json"
CONFIG_FILE = VIPERA_HOME / "config.json"
# Measured per-model/per-machine tuning profiles (see vipera.optimize.tuning).
TUNING_FILE = VIPERA_HOME / "tuning.json"

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8686,
    "default_model": None,
    # Aliases to serve together (lazy-loaded); None means single-model mode.
    "models": None,
    "max_loaded": 2,
    # `auto` uses llama-server when a CUDA build can see the card (its C++
    # engine is measurably faster than the Python binding), and falls back to
    # the in-process engine otherwise.
    "backend": "auto",
    # Parallel slots on the server backend. 1 keeps one context, which is what
    # a 4 GB card can hold; raise it for agent workloads with concurrent
    # requests (each slot gets ctx/parallel of the window).
    "parallel": 1,
    # HTTP layers to mount: `openai` for /v1/*, `ollama` for /api/*.
    "protocols": "openai",
    # Load the model with embedding support. Off by default: llama.cpp needs
    # the whole context built for it (--embeddings + pooling on the server
    # backend), which a chat-only run should not pay for.
    "embeddings": False,
    "profile": "auto",
    "ctx": 8192,
    "kv_cache": "q8_0",
    # flash_attn is deliberately absent: the planner derives it from kv_cache
    # (quantized KV requires it), so a config key would only look effective.
    "n_batch": 512,
    # 0 = auto (server backend uses 896 for MoE plans, min(n_batch, 512)
    # otherwise). -ub is the prefill lever; -b is raised to at least this.
    "n_ubatch": 0,
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
    # llama-server's own still-free speed lever: --spec-type ngram-*. The
    # in-process engine has no equivalent (llama-cpp-python's prompt-lookup
    # draft measured -29% decode on this box), so the value only applies to the
    # `server` backend. "none" turns it off.
    "spec_type": "ngram-map-k",
    # The optimization layer (vipera.optimize). `shaping` trims a request that
    # is larger than the loaded context instead of letting llama.cpp cut the
    # system prompt itself; `tuning` reuses the split a previous `vipera bench`
    # measured instead of re-deriving it from the VRAM formula.
    "shaping": True,
    "tuning": True,
}

_KV_CACHE_TYPES = {"f32", "bf16", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}

# "inprocess" drives llama-cpp-python in this process; "server" runs the
# llama-server binary as a child and talks OpenAI/SSE to it. Additional names
# can be installed as plugins (see vipera.backends.names).

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off", ""}

_TYPE_MAP = {
    "port": int,
    "embeddings": bool,
    "ctx": int,
    "n_batch": int,
    "n_ubatch": int,
    "max_loaded": int,
    "max_tokens": int,
    "parallel": int,
    "shaping": bool,
    "tuning": bool,
    "temperature": float,
    "top_p": float,
    "repeat_penalty": float,
}


def ensure_dirs() -> None:
    for d in (VIPERA_HOME, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _split_protocols(value) -> list[str]:
    """Normalize `protocols` from "a,b" or a JSON list into ordered names."""
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return []
    seen: list[str] = []
    for part in parts:
        name = str(part).strip().lower()
        if name and name not in seen:
            seen.append(name)
    return seen


def _coerce(cfg: dict) -> dict:
    for k, cast in _TYPE_MAP.items():
        if k in cfg and cfg[k] is not None:
            try:
                if cast is bool:
                    cfg[k] = str(cfg[k]).lower() in ("1", "true", "yes", "on")
                else:
                    cfg[k] = cast(cfg[k])
            except (ValueError, TypeError):
                # Never let a typo survive as a string: plan() would receive it
                # and fail with a TypeError far from the config file.
                print(
                    f"vipera: config {k}={cfg[k]!r} is not a valid {cast.__name__}; "
                    f"using {DEFAULT_CONFIG[k]!r}",
                    file=sys.stderr,
                )
                cfg[k] = DEFAULT_CONFIG[k]
    if cfg.get("kv_cache") not in _KV_CACHE_TYPES:
        cfg["kv_cache"] = DEFAULT_CONFIG["kv_cache"]
    if not _is_known_backend(cfg.get("backend")):
        cfg["backend"] = DEFAULT_CONFIG["backend"]
    if cfg.get("profile") not in profile_names():
        cfg["profile"] = DEFAULT_CONFIG["profile"]
    if cfg.get("parallel", 1) < 1:
        cfg["parallel"] = DEFAULT_CONFIG["parallel"]
    requested = _split_protocols(cfg.get("protocols"))
    known = [name for name in requested if _is_known_protocol(name)]
    for name in requested:
        if name not in known:
            print(
                f"vipera: unknown protocol {name!r}; valid protocols: "
                f"{', '.join(protocol_names())}",
                file=sys.stderr,
            )
    cfg["protocols"] = ",".join(known) or DEFAULT_CONFIG["protocols"]
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
    if "parallel" in cfg and int(cfg["parallel"]) < 1:
        raise ValueError("parallel: must be at least 1")
    if "backend" in cfg and not _is_known_backend(cfg["backend"]):
        valid = ", ".join(backend_names())
        raise ValueError(f"backend: must be one of {valid}")
    if "profile" in cfg and cfg["profile"] not in profile_names():
        valid = ", ".join(profile_names())
        raise ValueError(f"profile: must be one of {valid}")
    if "protocols" in cfg:
        requested = _split_protocols(cfg["protocols"])
        unknown = [name for name in requested if not _is_known_protocol(name)]
        if unknown or not requested:
            valid = ", ".join(protocol_names())
            found = ", ".join(unknown) or "(empty)"
            raise ValueError(f"protocols: unknown {found}; must be one of {valid}")
        cfg["protocols"] = ",".join(requested)
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
                merged = {**DEFAULT_CONFIG, **data}
                # A JSON null for a key with a real default is a typo, not a
                # value; letting it through crashed plan() with a TypeError.
                for key, default in DEFAULT_CONFIG.items():
                    if merged.get(key) is None and default is not None:
                        merged[key] = default
                return _coerce(merged)
            quarantine_corrupt(CONFIG_FILE, ValueError("not a JSON object"))
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    ensure_dirs()
    # Write-then-rename so a crash mid-write cannot leave a truncated config
    # that load_config() would silently replace with defaults.
    tmp = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, CONFIG_FILE)
