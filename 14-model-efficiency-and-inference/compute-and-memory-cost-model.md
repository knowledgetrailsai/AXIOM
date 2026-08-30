# Compute and Memory Cost Model

Architecture decisions should track more than parameter count. A lower theoretical FLOP count does not guarantee lower wall-clock latency — memory bandwidth, communication, and sequential depth all compete with raw arithmetic to determine actual serving cost.

## The standard FLOPs approximation

For a Transformer, a widely used rule of thumb approximates the forward-pass compute per token as:

```
FLOPs/token (forward) ≈ 2 × N
```

where N is the number of non-embedding parameters. The factor of 2 comes from each parameter participating in one multiply and one add per token in a matrix multiply. Training a token (forward pass plus backward pass, where the backward pass costs roughly twice the forward pass) is commonly approximated as:

```
FLOPs/token (full training step) ≈ 6 × N
```

**Worked training-cost example.** Take a 7×10⁹ parameter dense model trained on 1×10¹² (one trillion) tokens:

```
Total training FLOPs ≈ 6 × N × tokens
                     = 6 × 7×10⁹ × 1×10¹²
                     = 4.2×10²²  FLOPs
```

This is the standard way training compute budgets are estimated before a run: pick a target parameter count and token count, multiply by 6, and get a FLOPs budget that can be checked against available accelerator throughput and training time.

## Quantities beyond FLOPs

| Quantity | Why it matters |
|---|---|
| total parameters | Storage footprint and distributed placement — see the memory table in [quantization-and-architecture.md](quantization-and-architecture.md) |
| active parameters | Sets FLOPs per token/example directly, via the 2×N approximation above; for sparse models, active parameters can be far smaller than total (see [05-sparse-and-mixture-of-experts/dense-vs-sparse-computation.md](../05-sparse-and-mixture-of-experts/dense-vs-sparse-computation.md)) |
| activation memory | Training and prefill footprint — must be held in memory alongside weights during the forward (and backward, in training) pass |
| persistent inference state | KV cache, recurrent/SSM state, or explicit neural memory — the quantity that grows (or doesn't) with context length; see [kv-cache-and-memory-bandwidth.md](kv-cache-and-memory-bandwidth.md) |
| communication | All-reduce cost for data/tensor-parallel training, all-to-all cost for MoE dispatch (see [05-sparse-and-mixture-of-experts/moe-systems-tradeoffs.md](../05-sparse-and-mixture-of-experts/moe-systems-tradeoffs.md)) |
| sequential depth | Sets a latency floor independent of available parallel compute — a strictly sequential recurrence (see [06-state-space-and-recurrent-alternatives](../06-state-space-and-recurrent-alternatives/xlstm.md)) cannot be sped up just by adding more parallel accelerators |
| memory bandwidth | Often the actual bottleneck for autoregressive decode, as shown by the arithmetic-intensity argument in [kv-cache-and-memory-bandwidth.md](kv-cache-and-memory-bandwidth.md) |

## How architecture families differ on this model

- **Dense Transformer:** total ≈ active parameters, so the 2×N and 6×N approximations apply directly with N as the full parameter count; KV cache grows with context.
- **MoE:** total can greatly exceed active parameters (e.g. Mixtral 8x7B's ~47B total vs. ~13B active), so FLOPs approximations should use active parameters, not total; routing communication adds a cost with no dense-model analogue.
- **SSM/recurrent:** compact, fixed-size state avoids the growing-cache term entirely, but inference remains sequential step-by-step, imposing the sequential-depth latency floor above.
- **Diffusion:** total cost multiplies by the number of denoising steps — a single "sample" actually requires many forward passes through the denoiser (see [07-generative-model-architectures/diffusion-models.md](../07-generative-model-architectures/diffusion-models.md)).
- **World-model planning:** base per-step rollout cost is multiplied by however many candidate futures are simulated, before a plan is selected.

## The core caveat

None of these approximations account for how well a given workload maps onto specific hardware — a theoretically cheaper FLOPs count can still lose to a theoretically more expensive one if the cheaper model is memory-bandwidth bound, communication-bound, or limited by sequential depth in a way the more expensive model isn't. Use this table as a checklist of quantities to evaluate together, not as a single number to optimize in isolation.

## References

- Jiang, A.Q. et al. (2024). *Mixtral of Experts.* [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).

[Back to index](../INDEX.md)
