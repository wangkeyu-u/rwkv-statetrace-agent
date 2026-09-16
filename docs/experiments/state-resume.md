# state-resume

## Question and experiment
The controller experiment stops a Replay task, loads checkpoint history/state and resumes without repeating completed tools. FakeBackend serialization separately checks cursor/state and backend mismatch. API task-only storage is covered without a live endpoint. **Native RWKV recurrent-state restore and speedup: not run.**

## Reproduce

```bash
uv sync --extra dev
uv run python scripts/run_contract_experiments.py
```

## Results

[Machine-readable case outcomes](contract-results.json) record commands, exit codes, case names, environment and the preserved intentionally failing fixture output. Groups overlap: do not sum group counts as unique tests. Replay/fake-adapter tests are executable controls; no live API, model weights, neural memory quality, model latency or native performance is claimed.

## Limitations

No real native runtime is bundled. Generic CLI resume/fork does not launch arbitrary live continuation; use the controller and checkpoint manager integration. Future native evaluation must compare uninterrupted versus resumed outputs under the same model/runtime/precision and record saved-state size and prefix replay cost.
