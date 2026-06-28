import pytest

from src.backends import ApiFamily, Backend, BackendError, available_backends, resolve


def test_resolve_cpsat_is_integer_no_lazy() -> None:
    b = resolve("cp-sat")
    assert b.family is ApiFamily.CP_SAT
    assert b.continuous_flow is False
    assert b.supports_lazy is False  # CP-SAT uses the iterative cut loop
    assert b.solver_type is None


def test_resolve_mathopt_backends_are_continuous() -> None:
    for name in ("scip", "highs"):
        b = resolve(name)
        assert b.family is ApiFamily.MATH_OPT
        assert b.continuous_flow is True
        assert b.solver_type is not None


def test_resolve_unknown_name_raises() -> None:
    with pytest.raises(BackendError, match="unknown backend"):
        resolve("glpk")


def test_available_backends_always_includes_cpsat_scip_highs() -> None:
    names = {b.name for b in available_backends()}
    assert {"cp-sat", "scip", "highs"} <= names


def test_available_backends_returns_backend_objects() -> None:
    assert all(isinstance(b, Backend) for b in available_backends())


def test_resolve_gurobi_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.backends._gurobi_available", lambda: False)
    with pytest.raises(BackendError, match="unavailable"):
        resolve("gurobi")


def test_resolve_gurobi_available_returns_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.backends._gurobi_available", lambda: True)
    b = resolve("gurobi")
    assert b.name == "gurobi"
    assert b.family is ApiFamily.MATH_OPT


def test_scip_supports_lazy() -> None:
    assert resolve("scip").supports_lazy is True
    assert resolve("highs").supports_lazy is False
    assert resolve("cp-sat").supports_lazy is False
