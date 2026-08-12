# Limitations and safety boundary

StateTrace is a learning and demonstration project. The following limits are part of its design, not footnotes.

## Model limitations

- Small or untuned RWKV models may produce invalid JSON, select weak tools or stop with insufficient evidence.
- A fixed-shape recurrent state is compressed neural memory, not an unlimited or exact record of prior tokens.
- Restoring a state requires a compatible model, tensor layout, runtime and numeric precision. State files are not portable by assumption.
- Native continuation must feed only text not already represented by the restored state. Combining a continued state with a repeated full transcript double-encodes history; the current adapter contract leaves this runtime-specific delta boundary to the integrator.
- Sampling can make continued output differ after resume unless the runtime also restores every relevant random-number-generator state and decoding parameter.
- The Direct adapter contract has an incremental prompt boundary, but no particular third-party RWKV runtime ships in this repository; each adapter must verify its tokenizer, stop sequence, tensor serialization, and RNG restoration semantics.
- A standard OpenAI-compatible API normally hides recurrent tensors. API mode must not be presented as native state checkpointing.
- Replay mode is recorded behavior. Its checkpoint is a cursor for testing controller recovery, not an RWKV state and not live inference.

## Agent limitations

- Deterministic validation catches structural inconsistencies; it cannot prove that every diagnosis is semantically correct.
- File and line evidence can become stale if the workspace changes after an observation is recorded.
- Duplicate-call detection does not guarantee useful planning.
- The step and time limits may stop a task before completion.
- The bundled fixture is intentionally small and does not establish performance on large or adversarial repositories.
- State forking creates alternate continuations; it does not identify which branch is objectively best.

## Security limitations

- The tool layer reduces risk through path checks, allowlists, bounded output and subprocess timeouts, but it is not an OS-level sandbox.
- Tests execute repository code. A malicious repository can perform harmful actions even when the command itself is allowlisted.
- Only run the agent on trusted or properly isolated workspaces.
- Never expose secrets in the workspace, prompt, trace or generated report.
- Network isolation, containers, resource quotas and platform sandboxing are deployment responsibilities outside this prototype.
- The initial project diagnoses code and recommends changes; it does not need permission to write patches, commit, push or open pull requests.

## Performance claims

- Results from [Albatross](https://github.com/BlinkDL/Albatross) are implementation- and hardware-specific and are not inherited by this Python agent runtime.
- State size being independent of processed-token count does not make total checkpoint storage constant when many snapshots are retained.
- Resume-versus-replay measurements must report the exact model, hardware, precision, runtime, prompt and checkpoint interval.
- The [Qwen3.5 comparison runner](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py) includes hybrid architectures; simplistic “RWKV versus all Transformers” conclusions would be misleading.

## Compatibility and distribution

- Native model weights are not stored in this repository. Users must obtain weights and runtimes under their respective licenses and terms.
- Hardware-specific backends in [rwkv-mobile](https://github.com/MollySophia/rwkv-mobile) demonstrate broader deployment possibilities but are not integrated here.
- The dynamic [GitHub RWKV repository search](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories) is useful for discovery, not a stable specification or quality ranking.
