# Coding Standards

Adapted from the AI Blueprint Framework standards (`repo_template/standards/`) for this
project. The template targets full-stack web apps (FastAPI / Next.js / AWS); only the
Python-relevant rules apply here. This is a **scientific / optimisation reproduction**, so
the emphasis is on readable, explicit, whiteboardable formulations over web conventions.

---

## Tooling

| Tool | Purpose | Config |
|---|---|---|
| `uv` | Package management + running Python | `pyproject.toml` |
| `ruff` | Linting + formatting | `pyproject.toml` |
| `mypy` | Static type checking | `pyproject.toml` |
| `pytest` | Testing | `pyproject.toml` |

Always run Python through `uv` (`uv run python ...`, `uv run pytest`), never bare `python3`.

## Rules

- All functions and methods have type annotations — parameters and return types.
- No `# type: ignore` without an inline comment explaining why.
- No bare `except:` — catch specific exception types.
- Docstrings on public functions; for the formulation files, the docstring states the
  **math** (sets, variables, objective, constraints).
- Keep each formulation file (`direct_mip.py`, `benders.py`) self-contained and commented
  **at the constraint level**. Readability > cleverness — a non-specialist panel must be
  able to follow it.
- Secrets/licences via environment only (Gurobi license env vars) — never hardcode keys.
- Flag any formulation ambiguity resolved during implementation in an inline comment
  stating the assumption.
- Don't over-engineer: this is a demo, not a product. Three similar lines beat a premature
  abstraction.

## Naming

| Element | Convention | Example |
|---|---|---|
| Files | `snake_case` | `direct_mip.py` |
| Classes | `PascalCase` | `Instance`, `BendersResult` |
| Functions / methods | `snake_case` | `solve_direct_mip` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TIME_LIMIT` |
| Private helpers | `_snake_case` | `_build_residual_graph` |

## Tooling baseline (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true   # ortools ships partial / no stubs for some submodules
```

## Git

- Branches: `feat/`, `fix/`, `chore/`, `refactor/`, `docs/`, `test/` + short description.
- Commits: `<type>: <imperative, lowercase, no trailing period>`, subject ≤ 72 chars.
- Commit after each build stage (per the task spec) with a clear message.
- `main` is the integration branch; cut feature branches from it.
