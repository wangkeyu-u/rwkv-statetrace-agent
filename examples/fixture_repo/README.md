# Intentional date-boundary fixture

This tiny package is deliberately faulty. It is the trusted, deterministic workspace used by the StateTrace Replay demo and integration tests.

Run it from this directory:

```bash
python -m pytest -q
```

Expected baseline:

```text
1 passed, 3 failed
```

The passing case advances within a month. Month-end, year-end and leap-day cases fail because `next_calendar_day` resets the day to `1` while preserving the original month and year.

Do not fix the fixture on the default branch: its failure is an input to the diagnostic task, not a failing StateTrace product test. CI separately asserts that pytest exits with ordinary test-failure status `1` so import errors and collection errors cannot masquerade as the intended bug.
