# Validate all checkpoint artifacts before resume or fork

Recorded: 2026-09-15. This document retrospectively records the rationale behind the current implementation. Code-supported reasoning is an interpretation of the implementation, not a claim that an unrecorded historical experiment took place.

## Context
An intact model blob does not guarantee intact task metadata or trace.

## Options Considered

### Option A
Hash model_state.bin only. Pros: cheap. Cons: accepts mutated task/trace.

### Option B
Manifest hashes all durable artifacts. Pros: detects corruption and unexpected files. Cons: I/O scales with saved artifacts.

## Decision
Validate integrity, identity, backend and model before load/clone.

## Why
Fork must not convert a corrupt source into an apparently clean branch.

## Validation
[Fork experiment](../experiments/fork-behavior.md), `tests/test_checkpoints.py` tampering and clone cases.

## Trade-offs
Checksums detect corruption, not malicious rewriting by someone able to recompute the whole manifest; no external authenticity claim.

## What Would Change My Mind
A stronger threat model requiring signatures or an independently trusted append-only storage system.
