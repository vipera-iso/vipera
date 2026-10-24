import json
import types

import pytest

from vipera.optimize.tuning import Tuning, TuningStore, apply_tuning, fingerprint
from vipera.planner import Plan


def _hw(gpu="NVIDIA GeForce GTX 1650", vram=4.0, cores=4, ram=32.0, free=None):
    return types.SimpleNamespace(
        gpu_name=gpu,
        vram_total_gb=vram,
        vram_free_gb=free,
        cpu_cores=cores,
        ram_total_gb=ram,
    )


def _plan(**overrides) -> Plan:
    base = dict(
        gpu_layers=14,
        cpu_threads=4,
        context=8192,
        kv_cache="q8_0",
        flash_attn=True,
        n_batch=512,
        offload_kqv=True,
        use_mlock=False,
        note="Hybrid",
        n_layers=40,
    )
    base.update(overrides)
    return Plan(**base)


def test_fingerprint_separates_cards_and_hosts():
    assert fingerprint(_hw()) != fingerprint(_hw(gpu="NVIDIA GeForce RTX 4090", vram=24.0))
    assert fingerprint(_hw()) != fingerprint(_hw(cores=8))
    # Free VRAM is not part of it: it moves with whatever else is running.
    assert fingerprint(_hw()) == fingerprint(_hw(vram=4.0))


def test_store_round_trips_a_profile(tmp_path):
    store = TuningStore(tmp_path / "tuning.json")

    assert store.record("m", _hw(), gpu_layers=17, ctx=8192, gen_tps=6.2) is True

    profile = store.get("m", _hw())
    assert profile == Tuning(gpu_layers=17, ctx=8192, gen_tps=6.2, measured_at=profile.measured_at)
    assert store.get("other", _hw()) is None


def test_a_newer_measurement_is_kept_only_when_it_is_faster(tmp_path):
    path = tmp_path / "tuning.json"
    store = TuningStore(path)
    store.record("m", _hw(), gpu_layers=17, gen_tps=6.0)

    assert store.record("m", _hw(), gpu_layers=14, gen_tps=5.1) is False
    assert store.get("m", _hw()).gpu_layers == 17

    assert store.record("m", _hw(), gpu_layers=15, gen_tps=6.4) is True
    assert store.get("m", _hw()).gpu_layers == 15

    on_disk = json.loads(path.read_text())
    assert on_disk["_schema"] == 1
    assert on_disk["profiles"]["m"]


def test_a_measurement_without_a_rate_is_not_stored(tmp_path):
    store = TuningStore(tmp_path / "tuning.json")

    assert store.record("m", _hw(), gpu_layers=17, gen_tps=0.0) is False
    assert store.get("m", _hw()) is None


def test_a_corrupt_file_is_quarantined_not_overwritten(tmp_path, capsys):
    path = tmp_path / "tuning.json"
    path.write_text("{not json")

    store = TuningStore(path)

    assert store.get("m", _hw()) is None
    assert path.with_name("tuning.json.corrupt").exists()
    assert "unreadable" in capsys.readouterr().err


def test_apply_tuning_takes_the_measured_split(tmp_path):
    store = TuningStore(tmp_path / "tuning.json")
    store.record("m", _hw(), gpu_layers=17, gen_tps=6.2)

    p = apply_tuning(_plan(), "m", _hw(), store)

    assert p.gpu_layers == 17
    assert "measured: 17 layers, 6.20 t/s" in p.note
    # Everything else still comes from the planner.
    assert p.context == 8192
    assert p.kv_cache == "q8_0"


def test_apply_tuning_is_a_no_op_for_an_unmeasured_model(tmp_path):
    p = _plan()

    assert apply_tuning(p, "m", _hw(), TuningStore(tmp_path / "t.json")) is p


def test_apply_tuning_does_nothing_when_the_plan_already_agrees(tmp_path):
    store = TuningStore(tmp_path / "t.json")
    store.record("m", _hw(), gpu_layers=14, gen_tps=6.2)
    p = _plan(gpu_layers=14)

    assert apply_tuning(p, "m", _hw(), store) is p


def test_apply_tuning_clamps_to_the_model_layer_count(tmp_path):
    store = TuningStore(tmp_path / "t.json")
    store.record("m", _hw(), gpu_layers=99, gen_tps=6.2)

    assert apply_tuning(_plan(), "m", _hw(), store).gpu_layers == 40


def test_apply_tuning_leaves_a_moe_plan_alone(tmp_path):
    # A server plan offloads experts with -ncmoe; gpu_layers is not what it uses.
    store = TuningStore(tmp_path / "t.json")
    store.record("m", _hw(), gpu_layers=39, gen_tps=6.2)
    p = _plan(gpu_layers=14, moe_cpu_layers=12)

    assert apply_tuning(p, "m", _hw(), store) is p


def test_a_profile_does_not_apply_when_the_card_lost_room(tmp_path):
    # Measured with 3.6 GB free; another process now holds 1.5 GB. Replaying the
    # measured split would only fail and walk back down, so the formula wins.
    store = TuningStore(tmp_path / "t.json")
    store.record("m", _hw(free=3.6), gpu_layers=18, gen_tps=4.9)

    roomy = apply_tuning(_plan(gpu_layers=17), "m", _hw(free=3.5), store)
    tight = apply_tuning(_plan(gpu_layers=9), "m", _hw(free=2.1), store)

    assert roomy.gpu_layers == 18
    assert tight.gpu_layers == 9
    assert tight.note == "Hybrid"


@pytest.mark.parametrize("layers", [0, 1])
def test_measured_cpu_only_is_respected(tmp_path, layers):
    store = TuningStore(tmp_path / "t.json")
    store.record("m", _hw(), gpu_layers=layers, gen_tps=2.0)

    assert apply_tuning(_plan(), "m", _hw(), store).gpu_layers == layers


def test_a_profile_measured_at_a_smaller_context_does_not_apply(tmp_path):
    # An 8K split replayed into a 64K plan over-allocates the KV cache by 8x,
    # which is a failed context on a 4 GB card; the formula's split is honest.
    store = TuningStore(tmp_path / "t.json")
    store.record("m", _hw(), gpu_layers=18, ctx=8192, gen_tps=6.0)

    assert apply_tuning(_plan(context=65536, gpu_layers=9), "m", _hw(), store).gpu_layers == 9
    # Same or smaller than the measured window still fits.
    assert apply_tuning(_plan(context=4096, gpu_layers=9), "m", _hw(), store).gpu_layers == 18


def test_a_profile_without_a_recorded_context_still_applies(tmp_path):
    # Profiles written before ctx was stored have no window to compare against.
    store = TuningStore(tmp_path / "t.json")
    store.record("m", _hw(), gpu_layers=17, gen_tps=6.0)

    assert apply_tuning(_plan(context=65536), "m", _hw(), store).gpu_layers == 17
