# fork-behavior

## Question and experiment
The clone tests change task identity, preserve parent linkage and isolate branch mutation from the original. A corrupted source cannot be blessed by cloning; tampering in the clone is rejected. These are artifact/fake-adapter operations, not measured native state branching.

## Reproduce

```bash
uv sync --extra dev
uv run python scripts/run_contract_experiments.py
```

## Results

[Machine-readable case outcomes](contract-results.json) record commands, exit codes, case names, environment and the preserved intentionally failing fixture output. Groups overlap: do not sum group counts as unique tests. Replay/fake-adapter tests are executable controls; no live API, model weights, neural memory quality, model latency or native performance is claimed.

## Limitations

No real native runtime is bundled. Generic CLI resume/fork does not launch arbitrary live continuation; use the controller and checkpoint manager integration. Future native evaluation must compare uninterrupted versus resumed outputs under the same model/runtime/precision and record saved-state size and prefix replay cost.
