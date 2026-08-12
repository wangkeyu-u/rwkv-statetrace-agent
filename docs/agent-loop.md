# Agent loop and control boundary

## 1. What qualifies as the agent

StateTrace uses a strict feedback loop:

```text
goal + current observations
        ↓
model chooses one structured action
        ↓
controller validates and executes the action
        ↓
observation is recorded and returned to the model
        ↓
model continues, corrects, replans or submits a final report
```

This is different from a fixed workflow in which application code always runs `search`, then `read`, then `answer`. In StateTrace, application code defines the available actions and safety rules; the model selects the next tool and arguments from the current evidence.

The community [RWKV Agent Project Directory](https://agent.objects.rwkvos.com) is useful for this distinction: it separates strict agents, agent-related workflows, benchmarks and boundary projects. StateTrace targets the strict-loop category while borrowing benchmark ideas such as invalid-call recovery and multi-step observation feedback.

## 2. Structured action protocol

A typical action is one JSON object:

```json
{
  "type": "tool_call",
  "thought_summary": "The failing assertion identifies a boundary case; inspect its implementation.",
  "tool": "read_file",
  "arguments": {
    "path": "src/calendar_edge/dates.py",
    "start_line": 1,
    "end_line": 80
  }
}
```

`thought_summary` is a brief operational rationale. It is not a request for private chain-of-thought. The trace needs enough information for a reviewer to understand the transition without storing hidden reasoning.

A final action contains claims and evidence references. “Final” is a proposed completion, not unconditional success: the validator can reject it.

## 3. Responsibility split

| Model decides | Controller enforces |
| --- | --- |
| next tool | valid JSON/schema |
| tool arguments | known tool and typed arguments |
| whether evidence is sufficient | workspace path boundary |
| how to react to an error | command allowlist and timeout |
| when to propose completion | step/runtime/output limits |
| findings and recommendations | evidence and line-number checks |

This boundary preserves useful autonomy without delegating operating-system safety to sampled text.

## 4. Errors become observations

Recoverable failures are returned to the model as typed observations. Examples include:

- malformed JSON plus the expected schema;
- unknown tool plus the available tool list;
- invalid parameter plus its constraint;
- missing file;
- duplicate call warning;
- command timeout;
- final-report validation failure.

The controller does not silently replace a bad action with the action it thinks the model intended. That would make the apparent agent behavior misleading and the trace impossible to audit.

## 5. Evidence-driven completion

Each tool result receives an observation/evidence identifier such as `obs-004`. A final finding cites one or more identifiers. Deterministic validation checks at least that:

- cited observations exist;
- referenced workspace files exist;
- cited line numbers are in range;
- claimed test execution appears in the trace;
- the report does not claim success when the recorded exit status disagrees.

If validation fails, the failure list is appended as another observation and the model returns to the loop. The task only enters `COMPLETED` after validation succeeds.

## 6. Bounded autonomy

The default code-analysis toolset is intentionally narrow:

- list workspace files;
- search text/code;
- read bounded file ranges;
- run allowlisted test commands without a shell;
- perform safe arithmetic;
- propose the final report.

The sample task is read-only. StateTrace is not a security sandbox and should only inspect code the user has intentionally placed in scope. See [limitations](limitations.md).

## 7. Persistence layers

Three artifacts must not be conflated:

1. **Native RWKV recurrent state**: opaque/neural working memory exposed only by a compatible direct adapter.
2. **Task state and trace**: exact goal, tools, observations, evidence and lifecycle events.
3. **Replay cursor**: deterministic position in a recorded demonstration used by tests.

Direct mode can support all three where the runtime adapter implements state operations. API mode normally provides item 2 only. Replay provides items 2 and 3, never item 1.
