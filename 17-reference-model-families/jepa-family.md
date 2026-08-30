# JEPA Family

Architecture progression: **JEPA concept → I-JEPA → V-JEPA → V-JEPA 2**.

## Architectural lesson

The central idea is target prediction in representation space rather than raw reconstruction (see Latent Prediction): a context encoder and a target encoder produce embeddings of a context region and a target region, and a predictor is trained to produce the target embedding from the context embedding, never reconstructing raw pixels.

`L = distance(Predictor(Encoder_context(context)), Encoder_target(target))`

- **I-JEPA** (Assran et al., 2023) applies this to static images: mask out target blocks, encode the visible context, predict the masked blocks' representations.
- **V-JEPA** (Bardes et al., 2024) extends the idea to video, predicting masked spatio-temporal regions' representations rather than pixel-level future frames.
- **V-JEPA 2** further scales this video-prediction approach and connects it toward physical-world and planning applications.

This progression treats JEPA as an objective/architecture pattern that can wrap different backbone encoders (typically Vision Transformers), not a single fixed network — see the taxonomy distinction in `00-navigation-and-methodology/architecture-taxonomy.md`.

## Representative models

| Model | Modality | Backbone | Notable property |
|---|---|---|---|
| I-JEPA (Assran et al., 2023) | Static images | ViT | Block-masking, latent prediction, no pixel reconstruction |
| V-JEPA (Bardes et al., 2024) | Video | ViT | Spatio-temporal latent prediction |
| V-JEPA 2 (Meta AI, 2025) | Video | ViT | Scaled video latent prediction, world-model-oriented |

## References

- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf).
- Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
- Bardes, A. et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA).* [arXiv:2404.08471](https://arxiv.org/abs/2404.08471).
- Meta AI (2025). *V-JEPA 2.* [ai.meta.com/research/vjepa](https://ai.meta.com/research/vjepa/).

[Back to index](../INDEX.md)
