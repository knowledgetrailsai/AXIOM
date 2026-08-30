# Dense vs MoE

## Short Answer

A dense model runs every parameter for every token. A mixture-of-experts (MoE) model routes each token to a small subset of expert sub-networks, so total parameter count can be far larger than the compute cost per token. MoE only pays off when the serving system has enough batch size and communication bandwidth to keep those routed experts actually busy — otherwise the extra capacity sits idle while its communication overhead still has to be paid.

## Comparison

| Dimension | Dense | MoE |
|---|---|---|
| active parameters | all parameters run for every token | only the routed subset runs (e.g. top-2 of many experts) |
| total capacity | tied directly to active compute — you cannot grow one without the other | can be far larger than active compute, since most experts sit idle for any given token |
| execution pattern | regular, same compute path for every input | conditional — different tokens take different compute paths through different experts |
| communication | none beyond standard model/data parallelism | requires all-to-all communication to route tokens to the right expert's device |
| example scale | a 70B dense model uses ~70B parameters per token | Mixtral 8x7B has ~47B total parameters but only ~13B active per token (2 of 8 experts routed per token) |
| small-batch serving | strong — full utilization of every parameter on every request | can underutilize experts at small batch size, since not every expert gets enough tokens per batch to run efficiently |
| specialization | implicit, spread across shared weights | explicit — each expert can specialize on a different sub-distribution of inputs |

## The Real Trade-off

Mixtral 8x7B is a concrete illustration: 8 experts per MoE layer, each roughly 7B-parameter-sized, but only 2 are routed per token, giving ~13B active parameters per token against ~47B total parameters resident in memory. The model gets the quality benefit of ~47B parameters worth of learned specialization while paying the inference compute cost of only ~13B parameters per token. The cost that does not show up in that active-parameter number is communication: routing tokens to the correct expert, especially across devices, requires an all-to-all exchange whose overhead scales with how experts are distributed across the cluster. At small batch sizes, too few tokens route to any given expert to make that expert's matrix multiply efficient, so the sparsity gain shrinks or disappears.

## Hybrid Possibilities

Shared-expert designs keep some parameters dense (always active for every token) alongside routed sparse experts, capturing common computation once while still allowing specialization for the rest. Dynamic depth (see [adaptive-computation-and-dynamic-depth.md](../10-memory-and-adaptive-computation/adaptive-computation-and-dynamic-depth.md)) is a complementary axis: MoE changes *which* parameters run, dynamic depth changes *how many layers* run, and the two can be combined.

## References

- Jiang, A.Q. et al. (2024). *Mixtral of Experts.* [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
- Fedus, W., Zoph, B. & Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).

[Back to index](../INDEX.md)
