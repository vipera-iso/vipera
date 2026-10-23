import pytest

from vipera.errors import BackendError, PlannerError, ViperaError


def test_every_error_has_a_stable_code():
    assert ViperaError.code == "E0000"
    assert PlannerError.code == "E3001"
    assert BackendError.code == "E4001"


def test_planner_error_is_still_a_value_error():
    # Call sites written against 1.x catch ValueError; they must keep working.
    with pytest.raises(ValueError):
        raise PlannerError("unknown profile 'turbo'")


def test_backend_error_is_still_a_runtime_error():
    with pytest.raises(RuntimeError):
        raise BackendError("llama-cpp-python is not installed")


def test_errors_are_vipera_errors():
    with pytest.raises(ViperaError):
        raise PlannerError("x")
