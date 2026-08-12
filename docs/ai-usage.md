# How AI assistance was used

The assignment explicitly permits AI-assisted learning and development. This repository treats AI as an accelerator, while keeping claims and behavior verifiable.

## Assisted work

AI assistance was used to:

- compare the supplied RWKV and Agent references;
- turn the assignment into an implementation and acceptance checklist;
- draft module boundaries, tool schemas and failure cases;
- draft documentation and initial tests;
- review wording for unsupported claims, especially around native state, API mode and Replay mode;
- identify security tests for path traversal and command injection.

## Verification performed

Generated suggestions are not treated as evidence by themselves. The project checks them through:

- direct reading of the supplied upstream source and documentation;
- executable unit and integration tests;
- an intentionally failing fixture whose expected failures are checked in CI;
- checksums and metadata around checkpoint artifacts;
- deterministic validation of evidence identifiers, paths and line ranges;
- a Replay backend so controller behavior is reproducible without a downloaded model;
- a separate native adapter boundary so the repository does not invent an unsupported RWKV runtime API.

## Authorship boundary

The maintainers remain responsible for:

- selecting the project scope and safety boundary;
- reviewing and editing generated code and prose;
- confirming that commands operate only on intended paths;
- running the tests and interpreting failures;
- checking upstream licenses before copying or distributing third-party code;
- reporting experiment settings and limitations honestly.

No model weights, private prompts, credentials or third-party source files are generated into or redistributed with this repository.

## Sources consulted

- [RWKV Agent Project Directory](https://agent.objects.rwkvos.com)
- [RWKV-7 NumPy reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py)
- [RWKV-7 / Qwen3.5 NumPy runner](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py)
- [Albatross](https://github.com/BlinkDL/Albatross)
- [RWKV-7 training examples](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp)
- [DPLR mathematics](https://zhiyuan1i.github.io/posts/dplr-mathematics)
- [rwkv-mobile](https://github.com/MollySophia/rwkv-mobile)
- [RWKV.com project index source](https://github.com/BlinkDL/RWKV.com/blob/master/js/index.js)
- [GitHub RWKV repository search](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories)
