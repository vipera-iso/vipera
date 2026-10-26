"""Helpers for llama.cpp quantization, imatrix, and target-size planning.

The functions here build commands and make sizing decisions. They do not run
subprocesses; the CLI owns execution so it can stream progress and handle
interrupts.
"""

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_TOOL_NAMES = {
    "imatrix": "llama-imatrix",
    "quantize": "llama-quantize",
    "server": "llama-server",
    "cli": "llama-cli",
    "perplexity": "llama-perplexity",
    "bench": "llama-bench",
    "mtmd": "llama-mtmd-cli",
}

_QUALITY_ORDER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")

# Bytes-per-weight from gguf's block table: bytes * 8 / elements. Keep these
# exact (they drive --target-size math), not rounded to the nominal bit width.
_QUANT_BPW = {
    "NVFP4": 4.5,  # 36 B / 64 elems
    "MXFP4": 4.25,  # 17 B / 32 elems
    "TQ2_0": 2.0625,  # 66 B / 256 elems
    "TQ1_0": 1.6875,  # 54 B / 256 elems
    "Q1_0": 1.125,  # 18 B / 128 elems
    # Q2_0 is not in gguf 0.19 yet; this is the nominal 2-bit estimate.
    "Q2_0": 2.2,
    "Q8_0": 8.5,
    "Q6_K": 6.14,
    "Q5_K_M": 5.33,
    "Q5_K": 5.33,
    "Q4_K_M": 4.58,
    "Q4_K": 4.58,
    "Q3_K_M": 3.74,
    "Q3_K": 3.74,
    "Q2_K": 2.96,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}

_COMMON_BIN_DIRS = (
    Path.home() / ".vipera" / "bin",
    Path.home() / "llama.cpp" / "build" / "bin",
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
)


class OptimizerError(RuntimeError):
    """Raised when an optimization command cannot be planned or executed."""


@dataclass(frozen=True)
class QuantTarget:
    quant: str
    estimated_bytes: int
    target_bpw: float


def normalize_quant(quant: str) -> str:
    key = quant.strip().upper().replace("-", "_")
    aliases = {"Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M", "Q3_K": "Q3_K_M"}
    return aliases.get(key, key)


def _quant_bpw(quant: str) -> float:
    key = normalize_quant(quant)
    if key in _QUANT_BPW:
        return _QUANT_BPW[key]
    if key.startswith("IQ2"):
        return 2.3
    if key.startswith("IQ3"):
        return 3.3
    if key.startswith("IQ4"):
        return 4.3
    # Q8_K is not in the block table (only Q8_0 is), but it is still 8-bit;
    # without this it fell through to the Q4 fallback and halved the estimate.
    if key.startswith("Q8"):
        return 8.5
    if key.startswith("Q6"):
        return 6.14
    if key.startswith("Q5"):
        return 5.33
    if key.startswith("Q4"):
        return 4.58
    if key.startswith("Q3"):
        return 3.74
    if key.startswith("Q2"):
        return 2.96
    return 4.58


def _tensor_type_patterns(value: str | Sequence[str] | None) -> list[str]:
    """Validate ``pattern=type`` overrides, one per element."""
    if value is None:
        return []
    parts: Sequence[str] = [value] if isinstance(value, str) else value
    patterns: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        if "=" not in text:
            raise OptimizerError(f"invalid --tensor-type (expected PATTERN=TYPE): {text}")
        patterns.append(text)
    return patterns


def parse_prune_layers(value: str | Sequence[int] | None) -> list[int]:
    if value is None or value == "":
        return []
    parts: Sequence[str | int] = value.split(",") if isinstance(value, str) else value
    layers: list[int] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            layer = int(text)
        except ValueError as exc:
            raise OptimizerError(f"invalid layer number: {text}") from exc
        if layer < 0:
            raise OptimizerError(f"layer number must be non-negative: {layer}")
        if layer not in layers:
            layers.append(layer)
    return layers


def estimate_quant_for_target(
    current_size_bytes: int,
    current_quant: str,
    target_size_bytes: int,
) -> QuantTarget:
    """Pick the highest-quality known quant that should fit the target size.

    The estimate scales the current file size by bits-per-weight ratio. It is
    deliberately conservative and only covers common K-quants.
    """
    if current_size_bytes <= 0:
        raise OptimizerError("current model size must be greater than zero")
    if target_size_bytes <= 0:
        raise OptimizerError("target size must be greater than zero")

    current_bpw = _quant_bpw(current_quant)
    if target_size_bytes >= current_size_bytes:
        return QuantTarget(normalize_quant(current_quant), current_size_bytes, current_bpw)

    target_bpw = current_bpw * target_size_bytes / current_size_bytes
    for quant in _QUALITY_ORDER:
        bits = _quant_bpw(quant)
        if bits <= target_bpw:
            return QuantTarget(
                quant,
                int(current_size_bytes * bits / current_bpw),
                target_bpw,
            )

    return QuantTarget(
        "Q2_K",
        int(current_size_bytes * _quant_bpw("Q2_K") / current_bpw),
        target_bpw,
    )


def estimate_size_for_quant(
    current_size_bytes: int,
    current_quant: str,
    target_quant: str,
) -> int:
    if current_size_bytes <= 0:
        raise OptimizerError("current model size must be greater than zero")
    current_bpw = _quant_bpw(current_quant)
    return int(current_size_bytes * _quant_bpw(target_quant) / current_bpw)


def _is_runnable(path: Path) -> bool:
    if not path.is_file():
        return False
    if sys.platform == "win32":
        return path.suffix.lower() in (".exe", ".bat", ".cmd", "")
    return os.access(path, os.X_OK)


def find_llama_tool(tool: str, bin_dir: str | None = None) -> str:
    """Locate a llama.cpp tool by explicit directory, env var, PATH, or defaults."""
    primary = _TOOL_NAMES.get(tool, tool)
    names: list[str] = [primary]
    if tool not in names:
        names.append(tool)

    directories: list[Path] = []
    if bin_dir:
        directories.append(Path(bin_dir).expanduser())
    env_dir = os.environ.get("VIPERA_LLAMA_BIN")
    if env_dir:
        directories.append(Path(env_dir).expanduser())
    directories.extend(_COMMON_BIN_DIRS)

    for directory in directories:
        for name in names:
            for candidate in (directory / name, directory / f"{name}.exe"):
                if _is_runnable(candidate):
                    return str(candidate)

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    raise OptimizerError(
        f"{primary} was not found. Build llama.cpp, add it to PATH, set "
        "VIPERA_LLAMA_BIN, or pass --bin-dir."
    )


def tool_supports_flag(binary: str, flag: str) -> bool:
    try:
        result = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in f"{result.stdout}\n{result.stderr}"


def build_imatrix_command(
    binary: str,
    model: str | Path,
    calibration: str | Path,
    output: str | Path,
    ctx_size: int = 512,
    threads: int | None = None,
) -> list[str]:
    command = [
        binary,
        "-m",
        str(model),
        "-f",
        str(calibration),
        "-o",
        str(output),
        "-c",
        str(ctx_size),
    ]
    if threads:
        command.extend(["-t", str(threads)])
    return command


def build_quantize_command(
    binary: str,
    model: str | Path,
    output: str | Path,
    quant: str,
    *,
    imatrix: str | Path | None = None,
    prune_layers: str | Sequence[int] | None = None,
    prior_weight: int = 1,
    allow_requantize: bool = False,
    supports_prior_weight: bool = False,
    output_tensor_type: str | None = None,
    token_embedding_type: str | None = None,
    tensor_types: str | Sequence[str] | None = None,
    supports_tensor_type: bool = False,
) -> list[str]:
    command = [binary]
    if allow_requantize:
        command.append("--allow-requantize")
    if imatrix:
        command.extend(["--imatrix", str(imatrix)])

    # Mixed precision per tensor, the only lever that shrinks routed experts
    # without touching the attention blocks they sit beside. Measured on
    # LFM2.5-8B-A1B (GTX 1650, 64K): experts Q4_K -> Q3_K freed 1.2 GiB, which
    # the planner spent on five more expert layers (12 -> 17 of 24) and bought
    # +18.7% decode at the same VRAM (65.76 -> 78.05 t/s); PPL over a disjoint
    # eval slice moved 29.56 -> 28.69, inside the +/-1.8 error.
    for pattern in _tensor_type_patterns(tensor_types):
        if not supports_tensor_type:
            raise OptimizerError(
                "--tensor-type is not supported by this llama-quantize build; "
                "upgrade llama.cpp or omit the option."
            )
        command.extend(["--tensor-type", pattern])

    layers = parse_prune_layers(prune_layers)
    if layers:
        command.extend(["--prune-layers", ",".join(str(layer) for layer in layers)])

    if prior_weight != 1:
        if not supports_prior_weight:
            raise OptimizerError(
                "--prior-weight is not supported by this llama-quantize build; "
                "upgrade llama.cpp or omit the option."
            )
        command.extend(["--prior-weight", str(prior_weight)])

    # Mixed-precision: keep output layer and token embeddings at higher precision
    # while the rest of the model is quantized to `quant`.
    if output_tensor_type:
        command.extend(["--output-tensor-type", normalize_quant(output_tensor_type)])
    if token_embedding_type:
        command.extend(["--token-embedding-type", normalize_quant(token_embedding_type)])

    command.extend([str(model), str(output), normalize_quant(quant)])
    return command


def default_output_path(model: str | Path, quant: str) -> Path:
    path = Path(model).expanduser()
    suffix = normalize_quant(quant).lower()
    return path.parent / f"{path.stem}-vipera-{suffix}.gguf"


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OptimizerError(f"could not find a free output name for {path}")
