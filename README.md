# RWKV StateTrace Agent

**A traceable code-diagnosis agent whose model chooses each tool call, receives the observation, and decides what to do next. A native RWKV adapter can checkpoint, restore and fork recurrent working state.**

> Project status: an engineering demonstration, not a production security sandbox. The bundled Replay demo is deterministic recorded behavior—not live model inference. A standard text API is live inference but does not expose native RWKV recurrent state. Only a compatible Direct adapter may claim native state checkpointing.

## What it demonstrates

StateTrace accepts a goal such as:

```text
Run the tests, diagnose the date-boundary failures, cite the relevant source
and test lines, and recommend a minimal fix without modifying files.
```

The model—not a hard-coded workflow—selects a tool and its arguments. The controller validates that action, executes it inside a bounded workspace, assigns an evidence ID to the result, and feeds the observation back to the model. A proposed final report is checked against the trace before the task can complete.

```mermaid
flowchart TD
    A["User goal"] --> B["Model chooses structured action"]
    B --> C{"Valid action?"}
    C -- "No" --> D["Return protocol error as observation"]
    D --> B
    C -- "Yes" --> E["Execute bounded tool"]
    E --> F["Record result and evidence ID"]
    F --> G{"finish_report?"}
    G -- "No" --> H["Save task / supported model state"]
    H --> B
    G -- "Yes" --> I["Deterministic validation"]
    I -- "Rejected" --> J["Return missing or invalid evidence"]
    J --> B
    I -- "Accepted" --> K["Export Markdown / HTML report"]
```

The project focuses on the connection between a long-running Agent loop and RWKV-7's fixed-shape recurrent state. It preserves exact tool evidence separately because neural state is compressed working memory, not an auditable database.

## Three modes, three different claims

| Mode | Model decisions | Live generation | Checkpointed value | Native RWKV state? | Purpose |
| --- | --- | ---: | --- | ---: | --- |
| `rwkv-direct` | Compatible local RWKV adapter | Yes | Recurrent tensors plus task state | **Yes**, when the adapter exposes it | RWKV state experiments |
| `rwkv-api` | OpenAI-compatible RWKV endpoint | Yes | Task/trace only | **No** under the normal text API | Remote Agent execution |
| `replay` | Recorded action sequence | No | Replay cursor plus task state | **No** | CI and one-command demo |

Replay's cursor is checkpointable so recovery code can be tested. It must never be described as a neural checkpoint. Likewise, an API server's conversation ID is not evidence that native RWKV tensors were saved.

## Quick start

Requirements:

- Python 3.11 or later;
- `rg` (ripgrep) for `search_code`;
- a trusted local workspace; and
- no model download for Replay mode.

Clone this repository, then create an environment and install it in editable mode from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The bundled `demo` copies a packaged teaching fixture into `.statetrace/demo-workspace`, so it works from an editable checkout. A wheel installation needs the `demo` extra (`pip install 'rwkv-statetrace-agent[demo]'`) because the teaching run invokes pytest. The top-level `examples/` copy remains available for inspection.

Run the deterministic demonstration:

```bash
statetrace demo
```

The demo operates only on the copied teaching fixture under `.statetrace/demo-workspace`. Its inspectable source copy is `examples/fixture_repo`. The fixture contains an intentional month-boundary defect, so its own test suite should finish with `1 passed, 3 failed`. The Agent diagnoses it; it does not modify it.

Inspect generated task artifacts under `.statetrace/` and export a report if needed:

```bash
statetrace status --task-id <task-id>
statetrace export --task-id <task-id> --format html
```

Run project tests:

```bash
python -m pytest -q tests
```

The fixture failure is checked separately because it is expected:

```bash
cd examples/fixture_repo
python -m pytest -q
# expected: exit 1, with 1 passed and 3 failed
```

## Live execution

### OpenAI-compatible RWKV API

Copy `.env.example`, set endpoint variables in your environment, then run:

```bash
export RWKV_API_BASE_URL=http://127.0.0.1:8000/v1
export RWKV_API_MODEL=rwkv7
export RWKV_API_KEY='your-key-if-required'

statetrace run \
  --workspace examples/fixture_repo \
  --goal "Run the tests, diagnose the failures, and cite exact evidence." \
  --backend rwkv-api
```

API mode is live model inference. The common chat-completions protocol returns text, not recurrent tensors, so StateTrace does **not** advertise native state save/resume for this backend.

### Direct RWKV adapter (Python integration contract)

Direct mode is an adapter contract, not a bundled model runtime. Install a compatible native runtime and obtain model weights separately under their respective terms. The adapter must implement generation plus `save_state`, `load_state` and `clone_state`; StateTrace fails closed when those capabilities are missing.

The current CLI does not instantiate an arbitrary native runtime: `statetrace run` supports `rwkv-api`, while `rwkv-direct` deliberately exits with an explanation. Integrate Direct mode through Python by implementing `NativeRWKVAdapter`, constructing `RWKVDirectBackend(adapter)`, and passing it to `AgentController`. No weights are committed to this repository. See the contract in `src/statetrace/backends/direct.py`.

## Tools and safety controls

| Tool | Purpose | Important boundary |
| --- | --- | --- |
| `list_files` | inspect a sorted workspace tree | bounded depth/count; excluded internals |
| `search_code` | find text with `rg` | bounded results and output |
| `read_file` | read exact, numbered source ranges | no binary/out-of-root access; line cap |
| `run_tests` | run a supported test command | argv parsing, allowlist, no shell, timeout |
| `calculator` | deterministic arithmetic | AST/operator allowlist; no `eval` |
| `finish_report` | propose a cited diagnosis | rejected if evidence checks fail |

Allowlisting a test command does not make untrusted repository code safe: tests can execute arbitrary code. Run unfamiliar code in an OS/container sandbox. The prototype itself is not such a sandbox.

## Action and evidence protocol

A model emits one action at a time:

```json
{
  "type": "tool_call",
  "thought_summary": "The failures occur at boundaries; inspect the date helper.",
  "tool": "read_file",
  "arguments": {
    "path": "src/calendar_edge/dates.py",
    "start_line": 1,
    "end_line": 80
  }
}
```

The short `thought_summary` is an operational rationale, not private chain-of-thought. Malformed JSON, unknown tools and invalid arguments become observations with typed error codes; the model can correct itself on the next step.

Every executed tool result receives an ID such as `obs-003`. Final findings must cite existing evidence:

```json
{
  "file": "src/calendar_edge/dates.py",
  "line": 20,
  "claim": "The month-end branch preserves the original month and year.",
  "evidence_ids": ["obs-003", "obs-004"]
}
```

The validator checks referenced evidence, path existence, line ranges and recorded test execution. A failure returns to the Agent loop rather than silently accepting an unsupported answer.

## Persistence and recovery

After completed steps, the checkpoint manager stores exact task metadata and, when the backend supports it, the backend-owned state artifact:

```text
.statetrace/checkpoints/<task-id>/step_0004/
├── task_state.json
├── model_state.bin          # only when a supported state exists
├── model_state.meta.json
├── checksum.sha256          # only with model_state.bin
└── trace.jsonl
```

Resume and fork operations verify format, backend, model name and checksum before loading state:

```bash
statetrace resume --task-id <task-id>
statetrace fork --task-id <task-id> --from-step 4 --new-task-id alternative-search
```

`resume` is intentionally conservative: it reports already completed tasks without repeating actions. A stopped live task requires the original backend configuration, so generic CLI resume currently directs the caller to `CheckpointManager.load` and `AgentController` in Python. `fork` clones checkpoint artifacts; starting a new continuation also requires the corresponding backend/controller integration.

Saving many fixed-size neural states still uses storage proportional to the number of snapshots. StateTrace never calls the total checkpoint history “constant size.”

## Repository layout

```text
.
├── src/statetrace/
│   ├── backends/       # Direct, API and Replay boundaries
│   ├── tools/          # bounded code-analysis tools
│   ├── controller.py   # feedback loop / state machine
│   ├── checkpoints.py  # verified persistence and cloning
│   ├── protocol.py     # untrusted model-output parsing
│   ├── trace.py        # append-only JSONL events
│   └── validator.py    # deterministic final checks
├── examples/
│   ├── fixture_repo/   # intentionally faulty, trusted demo project
│   ├── replay_demo.json
│   └── tasks.json
├── tests/
├── docs/
└── .github/workflows/ci.yml
```

## Reading guide

- [RWKV-7 implementation notes](docs/rwkv7-notes.md): NumPy recurrence, TimeMix/ChannelMix and state semantics.
- [DPLR explained](docs/dplr-explained.md): diagonal decay, low-rank correction and affine composition.
- [Agent loop](docs/agent-loop.md): autonomy/control boundary, recovery and evidence.
- [Limitations](docs/limitations.md): model, security, performance and compatibility limits.
- [AI usage](docs/ai-usage.md): what AI assisted and how outputs were checked.

Primary references supplied with the assignment:

1. [RWKV Agent Project Directory](https://agent.objects.rwkvos.com)
2. [RWKV-7 NumPy reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py)
3. [RWKV-7 / Qwen3.5 NumPy runner](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py)
4. [Albatross](https://github.com/BlinkDL/Albatross)
5. [RWKV-7 training examples](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp)
6. [DPLR mathematics](https://zhiyuan1i.github.io/posts/dplr-mathematics)
7. [rwkv-mobile](https://github.com/MollySophia/rwkv-mobile)
8. [RWKV.com project index source](https://github.com/BlinkDL/RWKV.com/blob/master/js/index.js)
9. [GitHub RWKV repository search](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories)

## Honest interpretation of results

StateTrace can demonstrate that a particular direct runtime restored a recurrent checkpoint and avoided replaying a measured prefix. It does not establish lossless infinite memory, universal superiority over Transformer or hybrid architectures, or production-grade autonomous software engineering. Benchmark reports must name the exact model, runtime, hardware, precision and prompt.

## License

Project code and original documentation are available under the [MIT License](LICENSE). Model weights, runtimes and referenced projects keep their own licenses.
