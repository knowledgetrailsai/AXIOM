# CNN vs Transformer

## Short Answer

A CNN builds in locality: each layer only mixes information from nearby pixels, and the same filter is reused at every position. A Vision Transformer (ViT) builds in nothing about spatial structure — every patch can attend to every other patch from layer one — and has to learn locality from data instead. CNNs need less data to reach a given accuracy; ViTs can reach higher accuracy given enough data and pretraining.

## Comparison

| Dimension | CNN | Vision Transformer |
|---|---|---|
| core mechanism | convolution: shared local filter, weight-tied across positions | self-attention: every patch token attends to every other patch token |
| state/context | local receptive field, grows linearly with depth | global receptive field from the first layer |
| training parallelism | fully parallel over spatial positions | fully parallel over patch tokens |
| inference memory | none beyond activations, no growing cache | none for a single image; no KV cache needed since there is no autoregressive generation |
| sequence scaling | attention-free, so cost is linear in the number of pixels for a fixed filter size | self-attention cost is quadratic in the number of patch tokens: O(n²) for n patches |
| hardware profile | efficient on standard conv kernels; well optimized on essentially all hardware | benefits from the same matmul-heavy hardware as language Transformers, easing multimodal reuse |
| strongest regime | smaller labeled datasets, resource-constrained inference | large-scale pretraining data, especially when transferring across modalities |
| main limitation | receptive field only reaches global scale after many layers, limiting long-range interaction | patch-token attention cost grows quadratically with image resolution |

## The Real Trade-off

For an image split into a 14×14 patch grid (196 tokens, a common ViT-B/16 setting at 224×224 resolution), self-attention computes 196² ≈ 38,400 pairwise interactions per layer. Doubling resolution to 448×448 with the same patch size quadruples the token count to 784, and quadratic attention cost then grows by roughly 16×. A CNN's convolutional cost at the same resolution increase grows only proportionally to the number of pixels, not quadratically, because each filter only ever looks at a small fixed local window regardless of image size.

This is why CNNs remain competitive at high resolution and on smaller datasets: the inductive bias of locality does useful work for free that a ViT has to learn from data, and convolutional compute does not carry the quadratic attention penalty.

## Hybrid Possibilities

Hybrid designs use convolutional stems or local-window attention (as in Swin Transformer) to get local, cheap early-stage processing, then apply full or coarser-grained global attention in later stages — combining a CNN's cheap local structure with a Transformer's ability to model long-range interactions where it matters.

## References

- Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).
- Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* [arXiv:2103.14030](https://arxiv.org/abs/2103.14030).

[Back to index](../INDEX.md)
