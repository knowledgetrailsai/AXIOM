# Transformer vs SSM

## Short Answer

A Transformer keeps history as an explicit, growing set of key/value activations and attends over all of it. A selective state-space model (SSM) compresses history into a fixed-size state, updated with input-dependent gating. The real trade-off is addressable history versus learned compression: exact recall at growing cost, or compact state at the risk of losing something the compression discarded.

## Comparison

| Dimension | Transformer | Selective SSM (e.g. Mamba) |
|---|---|---|
| core mechanism | self-attention: `softmax(QK^T/√d)V` over all past positions | input-dependent linear recurrence: state update gated by the current input, not fixed transition matrices |
| state/context | explicit, addressable via attention weights | compressed into a fixed-size state vector |
| training parallelism | fully parallel via masked attention over the whole sequence | parallel via a hardware-aware parallel scan, not full attention-style parallelism, but still sub-linear-time-friendly in practice |
| inference memory | KV cache: O(n) in sequence length n | fixed-size state: O(1), independent of sequence length |
| sequence cost | attention is O(n²) in sequence length per layer (before efficient-attention variants) | linear O(n) in sequence length |
| exact retrieval | strong — attention can retrieve any past token directly if it learned to | depends entirely on whether the fixed-size state preserved the needed information; no exact-copy guarantee |
| streaming | possible with a sliding or capped cache, trading off exact long-range recall | natural — state size never grows, well suited to indefinite streams |

## The Real Trade-off

Concretely: attending over a 100,000-token context means storing and scanning roughly 100,000 key/value pairs per layer, with cost and memory scaling with that number. A selective SSM processing the same 100,000-token stream keeps a single state vector of fixed dimensionality (commonly in the hundreds to low thousands) throughout, no matter how long the stream runs. The Transformer can, in principle, recall the exact wording of token 3 out of 100,000 by attending directly to it. The SSM's ability to recall that same token depends entirely on whether its input-dependent gating decided that token's information was worth keeping in the compressed state — there is no architectural guarantee either way.

## Hybrid Possibilities

Attention-SSM hybrid architectures interleave a small number of attention layers (for exact, addressable retrieval where it matters) with a majority of SSM layers (for cheap linear-time processing of the bulk of the sequence), aiming to get most of the cost savings of linear recurrence with most of the retrieval strength of attention.

## References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Vaswani, A. et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

[Back to index](../INDEX.md)
