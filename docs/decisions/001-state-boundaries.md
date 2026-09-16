# Keep replay task state and native state distinct

Recorded: 2026-09-15. This document retrospectively records the rationale behind the current implementation. Code-supported reasoning is an interpretation of the implementation, not a claim that an unrecorded historical experiment took place.

## Context
`backends/direct.py` defines a runtime contract, while Replay stores a cursor and API mode has no recurrent tensors.

## Options Considered

### Option A
Serialize API conversation history and call it neural state. Pros: easy demo. Cons: changes the meaning of checkpoint.

### Option B
Declare backend capabilities explicitly. Pros: interpretable state semantics. Cons: native integration remains separate.

## Decision
Persist backend-owned state only where supported, alongside exact task/trace metadata.

## Why
A conversation ID and a Replay cursor cannot establish native recurrent-state restoration.

## Validation
[State experiment](../experiments/state-resume.md); `tests/test_checkpoints.py` tests fake state, not native tensors.

## Trade-offs
Native performance remains unmeasured; the CLI does not instantiate arbitrary direct runtimes.

## What Would Change My Mind
A compatible native adapter plus weights, hardware and measured no-replay restoration logs.
