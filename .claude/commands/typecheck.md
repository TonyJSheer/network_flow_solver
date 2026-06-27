Run the static type checker.

```bash
uv run mypy .
```

Report the results. If clean, print "✅ typecheck clean". If there are errors, show them.
Note: `ortools` ships partial/missing stubs for some submodules — `ignore_missing_imports`
is set, so a genuine error in our code is never hidden behind a missing-stub warning.
