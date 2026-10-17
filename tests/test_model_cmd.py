from vipera.commands.model_cmd import _read_meta


def test_read_meta_reads_quant_from_original_filename(tmp_path):
    model = tmp_path / "tmp-model.gguf"
    model.write_bytes(b"")

    meta = _read_meta(model, "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")

    assert meta["quant"] == "Q4_K_M"
    assert meta["size_bytes"] == 0
