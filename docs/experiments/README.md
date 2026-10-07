# Runtime contract experiments

Run from the repository root:

```bash
uv sync --extra dev
uv run python scripts/run_contract_experiments.py
```

[Machine-readable results](contract-results.json) record case names, commands, exit
codes, environment and the intentionally failing fixture output. Groups overlap;
do not add their counts as unique tests.


## State resume

Recorded result: **16/16 contract cases passed**.

The controller experiment stops a Replay task, loads checkpoint history/state and resumes without repeating completed tools. FakeBackend serialization separately checks cursor/state and backend mismatch. API task-only storage is covered without a live endpoint. **Native RWKV recurrent-state restore and speedup: not run.**


## Fork behavior

Recorded result: **2/2 contract cases passed**.

The clone tests change task identity, preserve parent linkage and isolate branch mutation from the original. A corrupted source cannot be blessed by cloning; tampering in the clone is rejected. These are artifact/fake-adapter operations, not measured native state branching.


## Evidence validation

Recorded result: **15/15 contract cases passed**.

Tests reject missing/out-of-range paths, unknown or wrong-file evidence, unverified test claims, contradictory test counts and malformed protocols. The intentionally faulty fixture remains 3 failed / 1 passed. Rejection tests passing means the guard worked, not that a live model learned to recover.


## Runtime boundary

Replay and fake adapters are executable controls. Live API inference, model
weights, neural memory quality, native recurrent tensors and speedup were not
measured. No native runtime is bundled. Generic CLI resume/fork does not launch
arbitrary live continuation; use the controller/checkpoint manager integration.
A future native evaluation must compare uninterrupted and resumed outputs under
the same model, runtime and precision, and record state size and prefix replay cost.
