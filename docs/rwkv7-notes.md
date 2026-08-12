# RWKV-7 implementation notes

This note explains the parts of RWKV-7 that directly motivate StateTrace Agent. It is a reading guide, not a substitute for the upstream implementation or paper.

## 1. The smallest useful reference implementation

[`rwkv_v7_numpy.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py) is the best starting point because it expresses one-token inference in ordinary NumPy and checks its result against the official runtime. The implementation exposes two kinds of recurrent state per layer:

1. the previous activation used by time mixing; and
2. a matrix state for every attention-like head.

The important property is structural: after the model shape and numeric precision are fixed, these tensors keep the same shape as more tokens arrive. They are updated rather than extended with one key/value entry per token.

The central matrix-state update can be read schematically as:

\[
S_t = S_{t-1} \odot w_t
      - (S_{t-1} k_t) (k_t \odot a_t)^T
      + v_t k_t^T
\]

and the current output reads from that state:

\[
y_t = S_t r_t
\]

The source code's exact orientation and broadcast shapes are authoritative. This schematic is intended to expose the roles:

- **decay** (`w`): attenuate selected old information;
- **state-dependent correction** (`k`, `a`): read and modify an existing direction;
- **write** (`v`, `k`): add information from the current token;
- **read** (`r`): retrieve the part needed for the current output;
- **gate** (`g`): control the contribution after the time-mixing operation.

These signals are input-dependent. RWKV-7 is therefore not a simple fixed exponential moving average.

## 2. TimeMix and ChannelMix

At a high level, an RWKV block has two recurrently friendly parts:

- **TimeMix** combines the current activation, the previous activation and the matrix state. It implements information flow across token positions.
- **ChannelMix** transforms features within a token and also mixes the current activation with a previous activation.

Both paths retain a small previous-activation state. TimeMix additionally owns the per-head matrix state. The actual upstream code should be consulted for layer normalization, low-rank parameterizations and exact projection order.

The symbols often seen in the v7 implementation are useful as a mental map, not independent API objects:

| Symbol | Operational intuition |
| --- | --- |
| `r` | read/query-like vector |
| `w` | data-dependent decay |
| `k` | addressing/write direction |
| `v` | value to write |
| `a` | state correction strength/direction |
| `g` | output gate |

The `train_temp` comments call `a` an in-context learning rate. That is a helpful intuition: the model dynamically controls how strongly the current token edits its internal state. It should not be confused with optimizer learning rate during weight training.

## 3. Fixed-shape state does not mean infinite perfect memory

For a fixed RWKV model, recurrent-state storage is independent of the number of tokens already processed. That is valuable for a persistent agent because a checkpoint does not need to contain a token-by-token KV history.

Several qualifications matter:

- Information from the past is compressed into finite tensors; it is not an exact transcript.
- Saving every checkpoint still consumes storage proportional to the number of checkpoints.
- Model weights and recurrent state are separate. A state can only be restored with a compatible model/runtime/precision contract.
- A state checksum proves file integrity, not semantic correctness.
- Exact facts such as tool arguments, file paths and evidence line numbers belong in structured task storage, not only in neural state.

StateTrace therefore treats the recurrent tensors as **neural working memory** and stores the auditable task trace separately.

## 4. Contrast with an append-only KV cache

[`run_rwkv7_qwen35.py`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py) is valuable because it places NumPy implementations behind a comparable interface. In its attention layers, a KV cache carries a sequence dimension which grows with cached context. In its recurrent layers, state is updated in place with a fixed model-dependent shape.

Qwen3.5 in that file is a hybrid rather than a straw-man “pure Transformer”: it combines recurrent/linear mechanisms with GQA layers. The correct lesson is not that every competing model has unbounded cache, but that different layer types have different state-growth and replay costs. Benchmarks must name the exact model, backend and cache policy.

## 5. Training and kernel engineering are separate layers

The [`train_temp`](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) directory shows that the recurrence equation alone is not a trainable model recipe. Initialization, parameter-specific learning rates, weight decay and custom forward/backward implementations all matter. StateTrace does not train a language model and makes no claim that an isolated TimeMix layer can simply be dropped into another network.

The [Albatross](https://github.com/BlinkDL/Albatross) repository explores high-performance inference using techniques such as custom CUDA kernels, CUDA Graphs, compilation and batched decode/prefill. It demonstrates that architecture-level opportunities still require serious systems work. StateTrace deliberately stays at the agent-runtime layer; it records latency and state size but does not reproduce Albatross performance claims.

## 6. Why this matters to the agent

In direct mode, StateTrace's backend boundary allows a compatible native adapter to return a new recurrent state with every generation. The persistence layer can then:

1. save that state alongside the completed action;
2. verify the state artifact and model metadata when resuming;
3. clone a checkpoint before exploring another plan; and
4. provide the state needed for a runtime-specific incremental continuation.

One implementation detail is essential: a host must not restore a state that already represents an earlier prompt and then feed that entire prompt through it again. It must either use native state plus only the new prompt delta, or discard the state and prefill the complete transcript. StateTrace marks Direct backends as `incremental_state`: the controller uses the full contract on the first turn and feeds only the newly produced Observation on subsequent stateful turns. API mode continues to use the bounded transcript because the normal text API exposes no recurrent tensor state.

This capability is intentionally not claimed for a standard OpenAI-compatible text API. Such an API normally exposes generated text, not RWKV's internal recurrent tensors. Replay mode's cursor is checkpointable for controller tests but is not a neural state.

## Further reading

- [RWKV-7 NumPy reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py)
- [RWKV-7 and Qwen3.5 NumPy runner](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py)
- [RWKV-7 temporary training code](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp)
- [Albatross inference experiments](https://github.com/BlinkDL/Albatross)
- [rwkv-mobile cross-platform runtime](https://github.com/MollySophia/rwkv-mobile)
