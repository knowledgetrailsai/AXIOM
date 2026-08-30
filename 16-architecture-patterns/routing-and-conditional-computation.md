# Routing and Conditional Computation

Routing answers: **which computation should run for this input?** Attention answers *which positions interact*; routing answers *which parameters execute at all*.

## The MoE routing formula

For a Mixture-of-Experts layer with `E` experts, a learned gate scores each expert for a token `x`:

`g(x) = softmax(W_g · x)`, a vector of length `E`

Only the top-`k` scoring experts actually run (commonly `k=1` or `k=2`), and their outputs are combined weighted by their (renormalized) gate scores:

`y = Σ_{i in top-k(g(x))} g_i(x) · Expert_i(x)`

Every other expert costs zero FLOPs for this token — this is what makes MoE a *sparse* scaling mechanism: total parameters (`E` experts' worth) can be far larger than active parameters (`k` experts' worth) per token. See `05-sparse-and-mixture-of-experts/mixture-of-experts.md` for the full worked treatment of MoE architecture.

## Other forms of routing

- **Layer skipping / adaptive depth**: a per-input decision about how many layers to run, rather than which parameters within a layer.
- **Specialist modules**: separate subnetworks for different input types (e.g. different modality encoders), selected by a hard rule rather than a learned gate.
- **Modality routing**: in multimodal models, deciding which expert or pathway handles which modality's tokens.

## Design questions

- **Load balance**: if the router concentrates traffic on a few experts, those experts become compute bottlenecks and the rest are undertrained — auxiliary load-balancing losses are typically added to the training objective to counteract this.
- **Differentiability**: hard top-k selection is not differentiable through the selection itself; gradients flow through the gate scores of the selected experts, not through the discrete choice of which experts were selected.
- **Batching and overflow**: real hardware batches tokens together, and if more tokens route to one expert than its capacity allows, excess tokens are dropped or overflow to a fallback — a systems-level consequence of a router-level decision.
- **Specialization stability**: whether experts converge to a stable, meaningfully different division of labor over training, versus route assignments that drift or degenerate (expert collapse, where most tokens route to a shrinking subset of experts).

[Back to index](../INDEX.md)
