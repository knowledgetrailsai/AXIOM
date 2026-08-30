# Predictive vs Generative World Models

## Short Answer

A latent predictive (JEPA-style) world model predicts a compact state vector for the future. A generative world model predicts the future observation itself — pixels or video. The latent approach is cheaper per rollout step and less directly checkable. The generative approach is more expensive per step and directly inspectable, because you can look at the predicted frame.

## Comparison

| Dimension | Latent predictive (JEPA-style) | Generative (Genie-style) |
|---|---|---|
| prediction target | embedding/state vector | pixels or video frame |
| output size per step | hundreds to low thousands of dimensions (e.g. a 768- or 1024-dim state vector) | a full frame, e.g. 256×256×3 = 196,608 raw values, or a compressed token grid of a few hundred discrete tokens |
| low-level detail | intentionally discarded | must be explicitly modeled to render a frame |
| inspectability | low — requires a probe or decoder to interpret | high — the output is directly viewable |
| rollout cost | one predictor forward pass over a small state vector per step | one full generative forward pass (often with a decoder) per step, decoding a full frame each time |
| planning signal | distance between predicted state and goal state in embedding space | must decode and inspect (or score) the rendered frame to compare against a goal |
| main risk | latent representation omits information the planner later needs | visually plausible frame that is causally or physically wrong |
| where it plans | in representation space, no rendering needed during search | either renders every candidate rollout, or attaches a separate scoring model to avoid that cost |

## The Real Trade-off

Concretely: a generative model rolling out 10 candidate action sequences over a 20-step horizon must produce 200 full frames if it evaluates each candidate by rendering. A latent predictive model performing the same search only propagates 200 state vectors of a few hundred to a couple thousand dimensions each — several orders of magnitude less data per rollout. That is why latent planning (as in V-JEPA 2, see [v-jepa-2.md](v-jepa-2.md)) is the cheaper choice for search-heavy planning, while generative world models (as in Genie, see [generative-world-models-and-genie.md](generative-world-models-and-genie.md)) are the right choice when the output itself — a viewable, playable environment — is the deliverable.

## Hybrid Possibilities

A system can plan cheaply in latent space using a predictive model, then decode a viewable frame only for the final chosen trajectory, or only when a human needs to inspect what the model expects to happen. This keeps search cost low while still producing an inspectable result when needed.

## References

- Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).
- Bruce, J. et al. (2024). *Genie: Generative Interactive Environments.* [arXiv:2402.15391](https://arxiv.org/abs/2402.15391).

[Back to index](../INDEX.md)
