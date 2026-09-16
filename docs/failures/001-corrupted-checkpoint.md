# Corrupted task metadata must not survive a fork

## Symptom
A model-state checksum alone leaves task metadata/trace mutation unchecked. The existing regression suite exercises this failure by changing task_state.json after save.

## Reproduction
`uv run pytest tests/test_checkpoints.py -k "tampered or corrupted or clone" -q`

## Root Cause
Task and trace are part of the trusted checkpoint, not auxiliary documentation.

## Attempts
Existing tests inject metadata/trace/state corruption and attempt resume/clone; this audit reruns them rather than inventing a historical incident.

## Final Fix
Current integrity manifest and clone prevalidation reject those artifacts. See [fork results](../experiments/fork-behavior.md).

## Remaining Risk
An attacker with write access can recompute unsigned hashes. This is an integrity mechanism for local artifacts, not remote attestation.
