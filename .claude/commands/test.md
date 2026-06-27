Run the full test suite.

```bash
uv run pytest
```

Report the results. If all pass, print "✅ tests passed". If any fail, show the failure
output. Remember: the toy instance has a known optimum, and the direct MIP and Benders must
agree on it — a mismatch there is a correctness bug, not a flaky test.
