# DPLR, from equations to agent state

This is a compact explanation of the mathematics used to reason about RWKV-7's recurrent update. The main reference is [DPLR 的数学与工程实现](https://zhiyuan1i.github.io/posts/dplr-mathematics/); exact RWKV expressions should be checked against the [NumPy reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py).

## 1. Diagonal plus low rank

A DPLR transition has the form

\[
P_t = D_t + b_t a_t^T
\]

where `D` is diagonal and `baᵀ` is low rank (rank one in this simplified expression). Applied to a matrix state:

\[
S_t = P_t S_{t-1} + k_t v_t^T
\]

Expanding the transition gives:

\[
S_t = D_t S_{t-1}
    + b_t(a_t^T S_{t-1})
    + k_t v_t^T
\]

This separates three useful operations:

1. `D S`: independently decay or retain state dimensions;
2. `b(aᵀS)`: read a projection of the old state and apply a low-rank correction across dimensions;
3. `kvᵀ`: write information from the current token.

The exact vector orientation varies with notation. The structural idea—diagonal evolution plus a cheap, input-dependent low-rank edit—is what matters here.

## 2. Why the low-rank term matters

A purely diagonal recurrence can remember or forget each dimension, but it cannot efficiently move information between directions. A low-rank term provides controlled cross-dimensional interaction without paying for an unrestricted dense transition at every token.

This gives a more useful state operation than accumulation alone: the model can use current input to decide which old feature to inspect and how to revise it. That is the connection behind phrases such as “state tracking” or “in-context learning” in RWKV-7 discussions. It is an analogy to online updating, not literal gradient descent on model weights during inference.

## 3. Sequential inference and parallel composition

A recurrence appears sequential because token `t` consumes the state from token `t-1`. However, a chunk's net effect can be summarized as an affine map:

\[
S_{out} = M S_{in} + B
\]

For two consecutive chunks:

\[
S_1 = M_0 S_0 + B_0
\]

\[
S_2 = M_1 S_1 + B_1
\]

Substitution yields:

\[
S_2 = (M_1M_0)S_0 + (M_1B_0 + B_1)
\]

So affine summaries compose associatively:

\[
(M_1,B_1) \circ (M_0,B_0)
= (M_1M_0,\; M_1B_0+B_1)
\]

Associativity enables scan/chunk strategies in training and prefill implementations. It does not make the naive Python token loop parallel by itself; efficient realization depends on the kernel and algorithm.

## 4. Connection to StateTrace Agent

An agent also evolves recurrently:

```text
previous working state + new tool observation -> updated working state
updated working state -> next action
```

The analogy is useful but should not be overstated:

- RWKV's recurrence happens inside a neural model at token granularity.
- Agent task state is an explicit program object containing goals, actions and evidence.
- The model state may compress prior observations, while the task trace preserves exact observations.
- DPLR does not itself implement planning, tool safety, validation or persistence.

StateTrace joins these layers deliberately. Direct mode checkpoints native recurrent tensors when a compatible adapter exposes them; the controller independently records exact evidence and state-machine events.

## 5. What the benchmark may and may not conclude

A fair resume-versus-replay experiment should report:

- model and checkpoint format;
- hardware, runtime and numeric precision;
- tokens already represented by the checkpoint;
- state artifact size;
- load and continuation latency;
- full-transcript prefill latency;
- whether outputs were generated with deterministic sampling.

It may conclude that native state restoration avoided reprocessing a particular prefix in that configuration. It may not conclude that state is lossless, that every Transformer behaves the same, or that an API session token is equivalent to RWKV recurrent tensors.
