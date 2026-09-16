# StateTrace operation and integration

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

For a wheel/package installation, include the Demo dependency:

```bash
python -m pip install 'rwkv-statetrace-agent[demo]'
```

The bundled `demo` copies a packaged teaching fixture into a new directory under `.statetrace/demo-workspaces/`, so repeated runs do not overwrite an earlier task's workspace. The `demo` extra installs pytest because the teaching run invokes it. The top-level `examples/` copy remains available for inspection in a source checkout.

Run the deterministic demonstration:

```bash
statetrace demo
```

The demo operates only on its copied teaching fixture under `.statetrace/demo-workspaces/`. Its inspectable source copy is `examples/fixture_repo`. The fixture contains an intentional month-boundary defect, so its own test suite should finish with `1 passed, 3 failed`. The Agent diagnoses it; it does not modify it. If pytest is absent, the CLI stops before creating a misleading “completed” diagnosis and prints the exact installation command.

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
  "evidence_ids": ["obs-003"]
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
├── integrity.sha256         # hashes every durable checkpoint artifact
└── trace.jsonl
```

Resume and fork operations verify the complete artifact manifest, format, task identity, backend, model name, state size and checksum before loading state:

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

