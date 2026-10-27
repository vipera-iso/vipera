import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

from vipera.backends.names import is_known as _is_known_backend
from vipera.backends.names import names as backend_names
from vipera.keepalive import parse_keep_alive
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
    # How long a pool-mode model stays loaded after its last request. Ollama
    # syntax: "5m", "1h", seconds, 0 to unload immediately, negative for
    # forever. Single-model mode keeps its engine for the process lifetime.
    "keep_alive": "5m",
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
    # Shared key every request must carry once set (`Authorization: Bearer` or
    # `x-api-key`); /health stays open for probes. None means no check, which is
    # what a loopback-only install wants and what every install did before this
    # key existed. Set it before binding to anything but 127.0.0.1.
    "api_key": None,
    # Browser origins allowed to call the gateway, as a comma-separated string
    # or a JSON list; "*" allows any. None installs no CORS middleware, so a
    # browser on another origin is refused by the browser, as documented.
    "cors": None,
    # Load the model with embedding support. Off by default: llama.cpp needs
    # the whole context built for it (--embeddings + pooling on the server
    # backend), which a chat-only run should not pay for.
    "embeddings": False,
    "profile": "auto",
    "ctx": 8192,
    "kv_cache": "q8_0",
    # Per-side overrides for -ctk/-ctv; None means "same as kv_cache". llama.cpp
    # takes the two halves separately, and they need not cost the same: K feeds
    # both the attention scores and the softmax denominator, V only the output.
    "kv_cache_k": None,
    "kv_cache_v": None,
    # `--no-kv-offload`: layer weights stay on the card while their KV cache
    # lives in system RAM. On a 4 GB card at long context the cache is most of
    # what keeps the layer count down, so this buys offload depth; it pays with
    # a PCIe read of the whole cache per token, which is why it is opt-in.
    "no_kv_offload": False,
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
    # llama-server's draft-free speculation: --spec-type ngram-*. Off by
    # default. The n-gram drafter only pays where the output repeats the input:
    # measured +37% on a prompt the model rewrites, -1% on 14B prose and -2.1%
    # on 0.5B prose (interleaved A/B, 3 runs each). A default has to hold for
    # both, and a rewrite workload can turn it on. Applies to the `server`
    # backend only; the in-process engine has no equivalent (llama-cpp-python's
    # prompt-lookup draft measured -29% decode on this box).
    "spec_type": "none",
    # Try a more aggressive MoE expert placement before the planner's own fit
    # and keep it if it loads: the fit comes from per-layer expert bytes plus a
    # measured overhead constant, so a card whose real ceiling is higher ends up
    # with expert layers in RAM that it could have computed. Costs up to two
    # extra spawns per start, and a probed split sits closer to the graph/compute
    # cliff than the planned one — turn it off to always start with the fit.
    "moe_probe": True,
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
    "no_kv_offload": bool,
    "moe_probe": bool,
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
                    # Strict, like the other types: a mistyped flag used to
                    # coerce to False and quietly turn a feature off.
                    cfg[k] = _parse_bool(cfg[k])
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
    for side in ("kv_cache_k", "kv_cache_v"):
        # Unlike kv_cache there is no default to fall back to: dropping a typo
        # to None hands the side back to kv_cache, and saying so is the only way
        # the user learns the flag was not applied.
        if cfg.get(side) is not None and cfg[side] not in _KV_CACHE_TYPES:
            print(
                f"vipera: config {side}={cfg[side]!r} is not a known KV cache type; "
                f"ignoring it (kv_cache={cfg['kv_cache']} still applies)",
                file=sys.stderr,
            )
            cfg[side] = None
    if not _is_known_backend(cfg.get("backend")):
        cfg["backend"] = DEFAULT_CONFIG["backend"]
    if cfg.get("profile") not in profile_names():
        cfg["profile"] = DEFAULT_CONFIG["profile"]
    if cfg.get("port") is not None and not _valid_port(cfg["port"]):
        print(
            f"vipera: config port={cfg.get('port')!r} is not in 1-65535; "
            f"using {DEFAULT_CONFIG['port']!r}",
            file=sys.stderr,
        )
        cfg["port"] = DEFAULT_CONFIG["port"]
    if cfg.get("parallel", 1) < 1:
        cfg["parallel"] = DEFAULT_CONFIG["parallel"]
    if cfg.get("api_key") is not None:
        # An empty string in the file is a half-configured key, not "open":
        # treating it as off is the only reading that cannot lock the owner out.
        cfg["api_key"] = str(cfg["api_key"]).strip() or None
    try:
        parse_keep_alive(cfg.get("keep_alive"))
    except ValueError as exc:
        print(
            f"vipera: config keep_alive={cfg.get('keep_alive')!r} is invalid "
            f"({exc}); using {DEFAULT_CONFIG['keep_alive']!r}",
            file=sys.stderr,
        )
        cfg["keep_alive"] = DEFAULT_CONFIG["keep_alive"]
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


def _valid_port(value) -> bool:
    """Whether ``value`` can be passed to a socket bind (and is not a bool)."""
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 65535


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
    for side in ("kv_cache_k", "kv_cache_v"):
        if cfg.get(side) is not None and cfg[side] not in _KV_CACHE_TYPES:
            valid = ", ".join(sorted(_KV_CACHE_TYPES))
            raise ValueError(f"{side}: must be one of {valid}")
    if "port" in cfg and not _valid_port(cfg["port"]):
        raise ValueError("port: must be an integer in 1-65535")
    if "parallel" in cfg and int(cfg["parallel"]) < 1:
        raise ValueError("parallel: must be at least 1")
    if "keep_alive" in cfg:
        try:
            parse_keep_alive(cfg["keep_alive"])
        except ValueError as exc:
            raise ValueError(f"keep_alive: {exc}") from exc
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
    if "api_key" in cfg and cfg["api_key"] is not None:
        if not isinstance(cfg["api_key"], str) or not cfg["api_key"].strip():
            raise ValueError("api_key: must be a non-empty string, or null to disable")
        cfg["api_key"] = cfg["api_key"].strip()
    if (
        cfg.get("cors") is not None
        and not isinstance(cfg["cors"], str)
        and not isinstance(cfg["cors"], (list, tuple))
    ):
        raise ValueError("cors: use a comma-separated string or a JSON list of origins")
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
    # The file can hold an API key, so it must not be world-readable. Written
    # before the rename: a crash leaves a tmp file nobody should read either.
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o600)
    os.replace(tmp, CONFIG_FILE)
