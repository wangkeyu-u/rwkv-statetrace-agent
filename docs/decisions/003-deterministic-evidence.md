# Reject unsupported model reports deterministically

Recorded: 2026-09-15. This document retrospectively records the rationale behind the current implementation. Code-supported reasoning is an interpretation of the implementation, not a claim that an unrecorded historical experiment took place.

## Context
A plausible final report can invent paths, evidence IDs or test outcomes.

## Options Considered

### Option A
Accept model prose when parsing succeeds. Pros: flexible output. Cons: unsupported diagnosis can pass.

### Option B
Check evidence IDs, files/line ranges and test output before finish. Pros: inspectable boundary. Cons: checks cannot establish semantic truth of every claim.

## Decision
Return validator rejection as an observation so the loop can recover.

## Why
Protocol validity and factual support are separate requirements.

## Validation
[Evidence experiment](../experiments/evidence-validation.md); `src/statetrace/validator.py` and `tests/test_validator.py`.

## Trade-offs
Trusted test commands still execute code; the tool allowlist is not an OS sandbox.

## What Would Change My Mind
A measured semantic verifier that improves correctness without accepting unsupported evidence or excessive false rejection.
