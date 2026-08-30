# Transformer vs SSM vs Recurrent Models

## Short Answer

Transformers keep every past token explicitly addressable through attention, at O(n²) compute in sequence length n and a KV cache that grows with context. Selective SSMs (Mamba) and modern recurrent models (RWKV, xLSTM) compress history into a fixed-size state and run in O(n), trading exact addressability for constant memory. The deeper trade-off is addressable history versus learned compression, not just quadratic versus linear complexity.

## Comparison

| Dimension | Transformer | Selective SSM / Mamba | RWKV / xLSTM-like |
|---|---|---|---|
| core mechanism | pairwise attention over all past tokens | input-selective linear state recurrence | decaying weighted sum (RWKV) or gated memory cell (xLSTM) |
| state/context | explicit KV cache, one entry per past token per head | compact fixed-size state per layer | compact fixed-size state per layer |
| training parallelism | fully parallel across sequence positions | parallel via selective-scan algorithm | parallel via decaying cumulative-sum form (RWKV); largely sequential (xLSTM) |
| inference memory | KV cache grows linearly with context length | constant, independent of context length | constant, independent of context length |
| sequence scaling | O(n²) compute, O(n) memory for the cache | O(n) compute and memory | O(n) compute and memory |
| hardware profile | dense matmuls, mature kernels and serving stack | needs a hardware-aware selective-scan kernel | needs custom kernels; less mature tooling |
| strongest regime | tasks needing exact recall of arbitrary earlier content | long-context streaming where compact state is worth more than exact recall | long-context streaming; xLSTM favors settings tolerating sequential training cost |
| main limitation | quadratic compute and growing cache at long context | history is compressed, not exactly addressable | history is compressed (RWKV, exponential decay) or training is sequential (xLSTM) |

**Worked complexity numbers.** Attention cost per layer scales as n² (pairwise comparisons); SSM/RNN cost per layer scales as n (one step per token). Comparing the operation count ratio n²/n = n at three context lengths:

| Sequence length n | Attention ops (∝ n²) | SSM/RNN ops (∝ n) | Ratio (n²/n = n) |
|---|---|---|---|
| 1,000 | 1,000,000 | 1,000 | 1,000× |
| 32,000 | 1,024,000,000 | 32,000 | 32,000× |
| 128,000 | 16,384,000,000 | 128,000 | 128,000× |

The ratio grows linearly with n itself — doubling context length doubles how many times more compute attention needs relative to a linear-time model. This is why the quadratic-vs-linear gap matters most exactly where long-context use cases live, and matters least at the short sequence lengths where most Transformer serving still operates today.

## The Real Trade-off

Complexity numbers explain *why* linear-time models get cheaper as context grows, but they don't explain what's lost. A Transformer's KV cache is, in effect, a complete transcript: any earlier token can be looked up exactly through attention. An SSM's or RNN's state is a *learned summary* of everything seen so far, sized once and never allowed to grow — new information can only enter by overwriting or blending with what's already there. For tasks that need to recall one specific, arbitrary fact from far back in the input, an explicit transcript is a structural advantage no amount of clever compression fully replaces. For tasks that need to track a compact evolving summary (e.g. a running state, a style, a topic) rather than exact facts, compression is not a disadvantage at all — it's the right shape for the problem.

## Hybrid Possibilities

Because the trade-off is about *which computation owns which job*, not about one mechanism being strictly better, hybrid architectures interleave attention layers (for exact addressability where it matters) with SSM or recurrent layers (for cheap compact context elsewhere) in the same stack. See [13-hybrid-architectures/transformer-plus-ssm.md](../13-hybrid-architectures/transformer-plus-ssm.md).

## References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Gu, A., Goel, K. & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces.* [arXiv:2111.00396](https://arxiv.org/abs/2111.00396).
- Peng, B. et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era.* [arXiv:2305.13048](https://arxiv.org/abs/2305.13048).
- Beck, M. et al. (2024). *xLSTM: Extended Long Short-Term Memory.* [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).

[Back to index](../INDEX.md)
