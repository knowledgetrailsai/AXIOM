# Architecture Taxonomy

An architecture is not one label. It is a point in a space of independent axes. Two models can share a "family name" and differ on every axis below, or carry different names and share most of them.

| Axis | Examples | Question |
|---|---|---|
| Connectivity | dense, local, sparse, routed | Which units can interact? |
| State | none, cache, recurrent, neural memory | What persists across steps? |
| Sequence mechanism | recurrence, attention, SSM | How does context propagate? |
| Generation | autoregressive, diffusion, flow | How is output sampled? |
| Prediction space | token/pixel, latent | What target is predicted? |
| Conditional compute | MoE, dynamic depth | Which parameters execute? |
| Modality | text, image, audio, video, action | What data enters/leaves? |
| Planning | none, search, rollout | Can candidate futures be evaluated? |

## A decision path for classifying an unfamiliar architecture

Ask these questions in order. Each answer narrows which sections of this repository are relevant.

1. **What does the sequence mechanism do with context?** Full pairwise access → attention (section 03). Compressed running state → recurrence/SSM (sections 02, and Mamba and SSM Families in 17). Fixed local window → convolution (section 02).
2. **Does every parameter run for every input, or only some?** All parameters, every input → dense. A learned subset per input → conditional compute / MoE (see Routing and Conditional Computation).
3. **Is output sampled step-by-step or refined iteratively?** Step-by-step, left-to-right → autoregressive (see Autoregressive Language Models). Iterative refinement from noise → diffusion. Continuous transport → flow-based.
4. **Is the training target raw content or a learned representation?** Raw pixels/tokens → standard reconstruction/prediction objectives. A target embedding, not raw content → latent prediction / JEPA-style (see Latent Prediction).
5. **Does the model only react, or does it also evaluate hypothetical futures?** No lookahead → standard feedforward/generative model. Explicit rollout or search over candidate future states → world model or planning system.

## Non-equivalent comparisons — a common classification error

- **"Transformer vs. MoE" is not a valid comparison.** MoE is usually a modification to the FFN sublayer inside a Transformer block, not an alternative to the Transformer itself. The valid comparison is "dense FFN vs. MoE FFN," with attention held constant.
- **"Transformer vs. JEPA" is not a valid comparison** for the same reason as above, in the other direction. JEPA is an objective/architecture pattern for target prediction (see Latent Prediction); it commonly *uses* a Transformer (specifically a ViT) as its encoder. The valid comparison is "reconstruction objective vs. latent-prediction objective," with the encoder backbone held constant.
- **World models are not one architecture.** Some are latent-predictive (JEPA-style, predicting future representations); others are generative (predicting future raw observations, e.g. via diffusion or autoregressive pixel/token prediction). Classify by prediction space (raw vs. latent) before comparing.
- **"Reasoning model" is a capability label, not an architecture label.** It typically mixes a backbone architecture choice, a post-training method (RL from verified outcomes, distillation from longer traces), and an inference-time strategy (extended chain-of-thought, search, self-consistency). Separate these three before making architectural claims.

[Back to index](../INDEX.md)
