# A failed overwrite removed the recoverable checkpoint

Recorded: 2026-10-01. These are persistence fault-injection tests, not native RWKV runtime measurements.

## Observed failure

The previous save path built a complete temporary directory, deleted the existing checkpoint with `shutil.rmtree`, then called `os.replace` to publish the new directory. If publication raised an `OSError`, exception cleanup removed the temporary directory too. An actual run with an injected publication error left `list_steps('task-safe') == []`: neither version remained available.

## Repair and trade-off

Keep the previous directory as `.step_<number>.backup` until the staged replacement is published. Ordinary publication errors restore the original directory. On the next list, load, save or clone, recovery restores that backup if the canonical directory is absent. If the new directory is present, recovery verifies its existing integrity manifest before deleting the backup. It does not recompute hashes or silently substitute an older version for a corrupted published checkpoint.

A second directory rename and temporary retention of the old generation cost I/O and storage. The manager assumes one writer per task. It does not lock concurrent writers or establish filesystem durability after sudden power loss. A process that exits during publication may leave an unused staging directory; such directories are excluded from step selection.

## Reproduce

```bash
uv sync --extra dev
uv run python -m pytest tests/test_checkpoints.py -k 'publish or overwrite or recovery' -v
```

The regressions cover four distinct outcomes:

- An injected publication `OSError` preserves every byte of the previous task, model state, trace and manifests.
- A subprocess exits with `os._exit` before publication. A fresh manager lists and loads the original checkpoint.
- An interruption after publication but before backup cleanup leaves the complete replacement selected on restart.
- A corrupted published directory with a retained backup fails integrity validation and keeps the backup available; recovery does not conceal the corruption.

The first three regressions failed against the previous implementation. Fake model state and task-only state exercise persistence contracts; they do not validate native tensor serialization.
