Run the linter and formatter. Ruff auto-fixes what it can.

```bash
uv run ruff check --fix .
uv run ruff format .
```

Report the results. If clean, print "✅ lint clean". If errors remain after auto-fix, show
them.
