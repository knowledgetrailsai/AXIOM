# Mixtral and MoE Families

Mixtral is a public reference for sparse **Transformer + MoE** design (see Mixture of Experts in section 05, and Routing and Conditional Computation).

## Architectural lesson

`attention → router → top-k expert FFNs → combine → residual stream`

Each Transformer block's dense FFN is replaced by `E` expert FFNs plus a learned router. For a token `x`, the router computes `g(x) = softmax(W_g · x)`, and only the top-`k` experts by score actually run:

`y = Σ_{i in top-k(g(x))} g_i(x) · Expert_i(x)`

Mixtral 8x7B: `E = 8` experts per MoE layer, `k = 2` — every token activates 2 of 8 experts, so total parameters (roughly 47B) substantially exceed active parameters per token (roughly 13B), since the attention layers and router are always active but only 2/8 of the expert FFN parameters run per token.

Study Mixture of Experts (section 05) for the general architecture; treat MoE as a scaling pattern applied inside a Transformer, not as a property unique to any one model family.

## Representative models

| Model | Experts per layer | Active experts (k) | Total params | Active params |
|---|---|---|---|---|
| Mixtral 8x7B (Jiang et al., 2024) | 8 | 2 | ~47B | ~13B |
| Mixtral 8x22B (Jiang et al., 2024) | 8 | 2 | ~141B | ~39B |

## References

- Jiang, A.Q. et al. (2024). *Mixtral of Experts.* [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* [arXiv:1701.06538](https://arxiv.org/abs/1701.06538).
- Fedus, W., Zoph, B. & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).

[Back to index](../INDEX.md)
