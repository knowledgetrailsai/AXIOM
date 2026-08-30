# Architecture Composition Patterns

Hybrid designs tend to fall into a small number of composition patterns, organized by which axis the mechanisms are split along.

## By Layer

`Embedding → SSM → SSM → Attention → MoE → SSM → Attention → Head`

Different block types alternate through the network's depth. Jamba (see [transformer-plus-ssm.md](transformer-plus-ssm.md)) is a concrete public example: mostly Mamba layers, with attention layers placed periodically, and MoE routing applied to some feed-forward sublayers.

## By Modality

`Vision front-end + Audio front-end + Text front-end → shared multimodal backbone`

Different mechanisms handle each modality's raw input before a shared backbone processes the resulting tokens together (see [08-multimodal-architectures/projection-and-cross-attention-fusion.md](../08-multimodal-architectures/projection-and-cross-attention-fusion.md) and [08-multimodal-architectures/native-multimodal-models.md](../08-multimodal-architectures/native-multimodal-models.md)).

## By Timescale

`Local context → attention → recurrent/neural long-term memory`

Short-range interaction is handled by one mechanism (e.g. attention within a local window), while longer-range persistence is handled by a separate mechanism built for that timescale (e.g. a recurrent or explicit memory module) rather than stretching one mechanism across both.

## By Function

`Generator → verifier → planner`

Different components handle generation, checking, and planning as distinct functional roles, rather than one model doing all three implicitly in a single forward pass. Speculative decoding (see [14-model-efficiency-and-inference/speculative-decoding.md](../14-model-efficiency-and-inference/speculative-decoding.md)) is a narrow instance of this pattern: a cheap draft model generates, a larger target model verifies.

## By Sparsity

`Shared dense expert + routed specialist experts`

Some MoE variants keep one expert always active (a shared/dense component providing a floor of common capacity) alongside routed experts that specialize, rather than routing every unit of feed-forward capacity.

## The design constraint

Composition should be justified by a specific, named bottleneck — compute, memory, exact retrieval, capacity, persistence, or latency. Adding a mechanism without a clear ownership boundary (which component is responsible for which part of the computation) usually increases implementation and tuning complexity faster than it increases capability; see [why-hybrid-architectures.md](why-hybrid-architectures.md) for the underlying design question this constraint follows from.

## References

- Lieber, O. et al. (2024). *Jamba: A Hybrid Transformer-Mamba Language Model.* [arXiv:2403.19887](https://arxiv.org/abs/2403.19887).
- Leviathan, Y., Kalman, M. & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).

[Back to index](../INDEX.md)
