# evidence-validation

## Question and experiment
Tests reject missing/out-of-range paths, unknown or wrong-file evidence, unverified test claims, contradictory test counts and malformed protocols. The intentionally faulty fixture remains 3 failed / 1 passed. Rejection tests passing means the guard worked, not that a live model learned to recover.

## Reproduce

```bash
uv sync --extra dev
uv run python scripts/run_contract_experiments.py
```

## Results

[Machine-readable case outcomes](contract-results.json) record commands, exit codes, case names, environment and the preserved intentionally failing fixture output. Groups overlap: do not sum group counts as unique tests. Replay/fake-adapter tests are executable controls; no live API, model weights, neural memory quality, model latency or native performance is claimed.

## Limitations

No real native runtime is bundled. Generic CLI resume/fork does not launch arbitrary live continuation; use the controller and checkpoint manager integration. Future native evaluation must compare uninterrupted versus resumed outputs under the same model/runtime/precision and record saved-state size and prefix replay cost.
