# tests/test_scaffold.py
def test_core_imports_available() -> None:
    import networkx  # noqa: F401
    import numpy  # noqa: F401
    from ortools.sat.python import cp_model  # noqa: F401

    assert cp_model.CpModel() is not None
