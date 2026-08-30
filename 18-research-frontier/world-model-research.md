# World Model Research

| Axis | Choices |
|---|---|
| prediction target | pixels/tokens (generative) vs latent embeddings (JEPA-style) |
| conditioning | passive observation only vs action-conditioned |
| horizon | short single-step prediction vs hierarchical multi-step planning |
| planning | none (pure prediction) vs MPC, tree search, or a learned policy consuming predictions |
| output | internal latent state only vs a rendered, viewable world |
| environment domain | video, physics simulation, robotics, or open-ended interactive worlds |

Two visible directions currently populate this space. Latent predictive models such as JEPA and V-JEPA 2 (see [v-jepa-2.md](../09-predictive-and-world-models/v-jepa-2.md)) predict compact state embeddings and plan by comparing them directly, without rendering. Generative interactive simulators such as Genie (see [generative-world-models-and-genie.md](../09-predictive-and-world-models/generative-world-models-and-genie.md)) generate literal, playable frames conditioned on inferred latent actions. See [predictive-vs-generative-world-models.md](../09-predictive-and-world-models/predictive-vs-generative-world-models.md) for the direct cost and inspectability comparison between the two.

## Open Question: Which Objective Learns Causal Structure

The central unresolved question in this research area is specific: does an objective that predicts well on held-out passive video (or held-out latent embeddings) actually learn dynamics that generalize to novel action sequences never seen during training, or does it only learn correlational structure that happens to match the training distribution's action patterns? A model can achieve low prediction error on data resembling training while still producing wrong predictions the moment a planner asks about a genuinely novel action sequence outside that distribution — and public research as of this writing does not settle, for any of the leading architectures in this space, how far that generalization actually extends.

## References

- Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).
- Bruce, J. et al. (2024). *Genie: Generative Interactive Environments.* [arXiv:2402.15391](https://arxiv.org/abs/2402.15391).

[Back to index](../INDEX.md)
