import pytest

from vipera.optimizer import (
    OptimizerError,
    build_imatrix_command,
    build_quantize_command,
    default_output_path,
    estimate_quant_for_target,
    estimate_size_for_quant,
    find_llama_tool,
    normalize_quant,
    parse_prune_layers,
    unique_output_path,
)


def test_estimate_quant_prefers_highest_quality_that_fits():
    choice = estimate_quant_for_target(
        current_size_bytes=4_800_000_000,
        current_quant="Q4_K_M",
        target_size_bytes=4_000_000_000,
    )

    assert choice.quant == "Q3_K_M"
    assert choice.estimated_bytes < 4_000_000_000


def test_estimate_quant_falls_back_to_q2():
    choice = estimate_quant_for_target(
        current_size_bytes=4_800_000_000,
        current_quant="Q4_K_M",
        target_size_bytes=1_000_000_000,
    )

    assert choice.quant == "Q2_K"
    assert choice.estimated_bytes < 4_800_000_000


def test_estimate_quant_keeps_current_when_target_is_larger():
    choice = estimate_quant_for_target(
        current_size_bytes=2_000_000_000,
        current_quant="Q4_K_M",
        target_size_bytes=8_000_000_000,
    )

    assert choice.quant == "Q4_K_M"
    assert choice.estimated_bytes == 2_000_000_000


def test_estimate_size_for_known_quant():
    size = estimate_size_for_quant(4_800_000_000, "Q4_K_M", "Q8_0")
    assert size > 4_800_000_000


def test_size_estimate_treats_q8_k_as_eight_bit():
    # Q8_K has no block-table entry, but it is still 8-bit. Falling through to
    # the generic Q4 fallback halved the estimate and the shrink recommendation.
    q8_k = estimate_size_for_quant(4_000_000_000, "Q4_K_M", "Q8_K")
    q8_0 = estimate_size_for_quant(4_000_000_000, "Q4_K_M", "Q8_0")
    assert q8_k == pytest.approx(q8_0)


def test_parse_prune_layers():
    assert parse_prune_layers("0, 2,2, 5") == [0, 2, 5]
    with pytest.raises(OptimizerError):
        parse_prune_layers("0,x")


def test_normalize_quant_aliases():
    assert normalize_quant("q4_k") == "Q4_K_M"
    assert normalize_quant("Q6_K") == "Q6_K"


def test_build_imatrix_command():
    command = build_imatrix_command(
        "llama-imatrix", "model.gguf", "cal.txt", "out.imatrix.gguf", 256, 4
    )
    assert command[:2] == ["llama-imatrix", "-m"]
    assert "-f" in command
    assert "-c" in command
    assert command[-2:] == ["-t", "4"]


def test_build_quantize_command_with_flags():
    command = build_quantize_command(
        "llama-quantize",
        "model.gguf",
        "out.gguf",
        "q4_k_m",
        imatrix="matrix.gguf",
        prune_layers="1,2",
        prior_weight=128,
        allow_requantize=True,
        supports_prior_weight=True,
    )

    assert "--allow-requantize" in command
    assert command[command.index("--imatrix") + 1] == "matrix.gguf"
    assert command[command.index("--prune-layers") + 1] == "1,2"
    assert command[command.index("--prior-weight") + 1] == "128"
    assert command[-3:] == ["model.gguf", "out.gguf", "Q4_K_M"]


def test_build_quantize_rejects_unsupported_prior_weight():
    with pytest.raises(OptimizerError, match="prior-weight"):
        build_quantize_command(
            "llama-quantize",
            "model.gguf",
            "out.gguf",
            "Q4_K_M",
            prior_weight=128,
            supports_prior_weight=False,
        )


def test_find_llama_tool_uses_bin_dir(tmp_path):
    tool = tmp_path / "llama-quantize"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(tool.stat().st_mode | 0o111)

    assert find_llama_tool("quantize", str(tmp_path)) == str(tool)


def test_find_llama_tool_maps_cli_to_llama_cli(tmp_path):
    tool = tmp_path / "llama-cli"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(tool.stat().st_mode | 0o111)

    assert find_llama_tool("cli", str(tmp_path)) == str(tool)


def test_find_llama_tool_reports_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("VIPERA_LLAMA_BIN", raising=False)
    monkeypatch.setattr("vipera.optimizer._COMMON_BIN_DIRS", ())
    monkeypatch.setattr("vipera.optimizer.shutil.which", lambda name: None)

    with pytest.raises(OptimizerError, match="was not found"):
        find_llama_tool("quantize", str(tmp_path))


def test_default_and_unique_output_paths(tmp_path):
    model = tmp_path / "model.gguf"
    default = default_output_path(model, "q4_k_m")
    assert default.name == "model-vipera-q4_k_m.gguf"

    default.write_bytes(b"x")
    unique = unique_output_path(default)
    assert unique.name == "model-vipera-q4_k_m-2.gguf"
    assert unique.exists() is False


def test_size_estimate_uses_block_table_bpw_for_2026_quants():
    # bytes-per-weight, not the nominal bit width: 36 B / 64 elems = 4.5.
    nvfp4 = estimate_size_for_quant(
        current_size_bytes=4_800_000_000, current_quant="Q4_K_M", target_quant="NVFP4"
    )
    assert nvfp4 == pytest.approx(4_800_000_000 * 4.5 / 4.58)

    q1_0 = estimate_size_for_quant(
        current_size_bytes=4_800_000_000, current_quant="Q4_K_M", target_quant="Q1_0"
    )
    assert q1_0 == pytest.approx(4_800_000_000 * 1.125 / 4.58)
    assert q1_0 < 4_800_000_000 * 0.3


def test_estimate_quant_from_nvfp4_current_size():
    choice = estimate_quant_for_target(
        current_size_bytes=4_800_000_000,
        current_quant="NVFP4",
        target_size_bytes=6_000_000_000,
    )
    # Target is larger than the file, so the current quant is kept and sized by
    # the block-table bpw (4.5) rather than the generic Q4 fallback (4.58).
    assert choice.quant == "NVFP4"
    assert choice.target_bpw == pytest.approx(4.5)
