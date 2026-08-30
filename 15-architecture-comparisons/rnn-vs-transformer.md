# RNN vs Transformer

## Short Answer

An RNN compresses all history into one fixed-size hidden state, updated one step at a time. A Transformer keeps every past token's activations explicitly and attends over them directly. The RNN's inference memory never grows; the Transformer's KV cache grows linearly with sequence length. The RNN trains sequentially; the Transformer trains in parallel across the whole sequence at once.

## Comparison

| Dimension | RNN/LSTM | Transformer |
|---|---|---|
| core mechanism | recurrent update: `h_t = f(h_{t-1}, x_t)` | self-attention over all previous positions |
| state/context | compressed into one fixed-size hidden state | explicit, addressable context via attention |
| training parallelism | sequential across timesteps — step t needs h_{t-1} | fully parallel across all sequence positions (with causal masking) |
| inference memory | fixed-size hidden state, O(1) regardless of sequence length | KV cache grows O(n) with sequence length n |
| sequence scaling | O(1) compute per new token | O(n) compute per new token due to attending over the full cache (naively; less with efficient attention) |
| long-range path | information must pass through every intermediate hidden-state update, n sequential steps to connect token 1 and token n | direct attention edge between any two tokens, one step regardless of distance |
| streaming | natural — process one token, update state, discard input | requires managing a growing or windowed KV cache |

## The Real Trade-off

For a token 1,000 positions back, an RNN's hidden state has been overwritten by 1,000 subsequent updates by the time it reaches the current step — whether that specific piece of information survives depends entirely on what the recurrence chose to keep, with no guarantee. A Transformer's attention can retrieve that same token's activation directly, in one step, regardless of distance, as long as it is still inside the KV cache. This is the core trade-off: RNNs pay O(1) memory for potentially lossy long-range recall; Transformers pay growing O(n) memory for exact addressable recall.

## Hybrid Possibilities

Selective SSMs (see [transformer-vs-ssm.md](transformer-vs-ssm.md)) modernize the RNN idea with input-dependent state updates and hardware-efficient parallel scans, narrowing the training-parallelism gap while keeping O(1) inference memory. Hybrid architectures interleave a few attention layers (for exact recall) with many linear-recurrence layers (for cheap long-context throughput).

## References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Vaswani, A. et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

[Back to index](../INDEX.md)
