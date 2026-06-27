Inspect an installed package to understand its exact API surface (useful for `ortools`
CP-SAT / MathOpt and `networkx` — where guessing the API wastes time).

Package: $ARGUMENTS

## Step 1 — Find the installed version

1. Read `pyproject.toml` — find the package in `[project.dependencies]` or the dev group.
2. Read `uv.lock` — find the pinned version (`version = "x.y.z"` under the package). The
   lock file is ground truth — use it, not the constraint in `pyproject.toml`.

## Step 2 — Locate the installed source

Glob inside the venv (package names use underscores on disk):
```
.venv/lib/python*/site-packages/<package>/
.venv/lib/python*/site-packages/<package>-*.dist-info/
```
Try both the PyPI name and the import name if the first glob finds nothing
(e.g. `ortools` imports as `ortools.sat.python.cp_model` / `ortools.math_opt.python` —
check the relevant submodule `__init__.py` for exports).

## Step 3 — Read the API surface (best signal first)

1. **Type stubs**: glob `<site-packages>/<pkg>/**/*.pyi` — most precise declarations.
2. **`__init__.py`**: top-level exports.
3. **Relevant submodule `__init__.py`**: for large packages.
4. **`METADATA`** in `.dist-info/`: confirms version, entry points.

Prefer `context7` MCP for current upstream docs when the installed source is thin.

## Step 4 — Report

```
Package:  <name>
Version:  <exact installed version>
Location: <path to installed source>

API surface:
<key classes, functions, constants — with signatures where visible>

Notes:
<version-specific behaviour, deprecated APIs, import paths to use>
```

If not installed, say so and suggest `uv add <package>`.
