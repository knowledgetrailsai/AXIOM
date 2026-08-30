# Post-Transformer Directions

"Post-Transformer" does not mean the Transformer disappears. Active research directions extend, replace, or hybridize specific pieces of the Transformer, not the entire paradigm:

- selective SSMs (Mamba-style linear-time recurrence, see [transformer-vs-ssm.md](../15-architecture-comparisons/transformer-vs-ssm.md)) as a replacement for attention's quadratic cost;
- RWKV and xLSTM-style modern recurrence, aiming for Transformer-level quality with recurrent-style constant inference memory;
- efficient/linear attention approximations that keep the attention mechanism but reduce its quadratic cost;
- attention-SSM hybrids, mixing a minority of attention layers with a majority of linear-recurrent layers;
- test-time neural memory (Titans, see [titans-test-time-memory.md](../10-memory-and-adaptive-computation/titans-test-time-memory.md)), adding memory that updates during inference itself;
- MoE and modularity, decoupling total parameter count from per-token compute cost (see [dense-vs-moe.md](../15-architecture-comparisons/dense-vs-moe.md));
- tokenizer-free byte/patch models, removing the fixed vocabulary (see [tokenizer-free-and-byte-level-models.md](tokenizer-free-and-byte-level-models.md));
- latent/recurrent reasoning, moving intermediate computation out of generated tokens (see [latent-reasoning.md](../11-reasoning-oriented-architectures/latent-reasoning.md));
- world models and action-conditioned predictors, extending prediction beyond next-token language modeling into embodied and physical domains (see [what-is-a-world-model.md](../09-predictive-and-world-models/what-is-a-world-model.md)).

## The Likely Shape of What Comes Next

A single universal replacement for the Transformer, matching or beating it on every axis at once, has not appeared in public research as of this writing. The more likely direction is a heterogeneous architecture: attention used specifically where its exact, addressable long-range recall earns its quadratic cost, and state-space recurrence, routing, or memory handling the rest of the sequence more cheaply. Which specific mix of these pieces ends up dominant, and at what model scale each trade-off tips one way or another, is an open empirical question, not a settled one.

## References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Behrouz, A., Zhong, P. & Mirrokni, V. (2025). *Titans: Learning to Memorize at Test Time.* [arXiv:2501.00663](https://arxiv.org/abs/2501.00663).
- Pagnoni, A. et al. (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens.* [arXiv:2412.09871](https://arxiv.org/abs/2412.09871).

[Back to index](../INDEX.md)
