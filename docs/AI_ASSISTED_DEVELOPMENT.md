# AI-assisted development

The [earlier AI usage record](ai-usage.md) describes the project's original assistance boundaries. Codex assisted this revision's contract runner, execution and documentation.

## Evidence from this revision

- The runner executes existing recovery, fork and evidence-validation tests and records named cases in [contract-results.json](experiments/contract-results.json).
- Replay and fake backends exercise controller behavior. They do not validate real native RWKV tensors or establish an inference speedup.
- The intentional teaching fixture still fails with three failures and one pass. The agent diagnoses that fixture; it does not silently repair it to make the demonstration pass.

The developer owns acceptance of the action protocol, persistence semantics and execution boundary. Live API inference and native runtime behavior remain unmeasured in this revision.
