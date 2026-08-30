# Vision Transformers

## Context and Plain-Language Explanation

A Vision Transformer (ViT) cuts an image into fixed-size patches, flattens and linearly projects each patch into a token, adds position information, and runs the result through a standard Transformer encoder. The image becomes a sequence, and the Transformer treats it exactly like a sequence of word tokens.

## Problem It Tries to Solve

CNNs bake in a locality prior — a pixel only directly interacts with nearby pixels within each layer's receptive field, and global relationships require depth to accumulate. Attention can model relationships between any two positions in one layer, so applying it to images lets the model learn which spatial relationships matter, rather than assuming only local ones do a priori.

## Core Architectural Idea

### Patch embedding math

An image of size `H × W × C` (height, width, channels) is split into `N` non-overlapping patches of size `P × P`:

`N = (H * W) / P^2`

Each `P × P × C` patch is flattened into a vector of length `P^2 * C` and linearly projected into the model's embedding dimension `D`:

`patch_embedding = Flatten(patch) · W_proj`, where `W_proj` has shape `(P^2 * C) × D`

**Worked example.** A 224×224×3 image (`H=W=224`, `C=3`), patch size `P=16`:

```
N = (224 * 224) / 16^2 = 50176 / 256 = 196 patches

Each patch flattens to a vector of length 16*16*3 = 768

Projection matrix W_proj: 768 x D  (e.g. D=768 for ViT-Base -> a 768x768 matrix)
```

The 196 resulting patch tokens (plus, in the original ViT, one extra learnable `[CLS]` token for classification) form a sequence of length 197 fed into a standard Transformer encoder. Position embeddings (learned, one per patch position) are added before the first block, exactly as in a text Transformer.

## Information Flow

```mermaid
flowchart LR
    Img[Image H x W x C] --> Split[Split into N patches of P x P]
    Split --> Flat[Flatten each patch]
    Flat --> Proj[Linear projection to dimension D]
    Proj --> Pos[Add position embedding]
    Pos --> Enc[Standard Transformer encoder]
    Enc --> Out[Classification / feature output]
```

## Components

| Component | Role |
|---|---|
| Patchify + linear projection | Converts a 2D image into a 1D sequence of embedding vectors, matching Transformer input format |
| Position embedding | Injects 2D spatial position information the patch sequence would otherwise lack |
| `[CLS]` token | An extra learnable token whose final representation is used for image-level classification |
| Standard Transformer encoder | Identical bidirectional self-attention + FFN stack used for text (see Transformer Block) |

## Computational Characteristics

| Dimension | Notes |
|---|---|
| Training parallelism | Fully parallel across patches — patchify and projection are trivially parallel; attention parallelizes over the patch sequence |
| Sequence scaling | Attention cost is `O(N^2)` in patch count `N`; since `N` scales as `1/P^2`, halving patch size quadruples `N` and roughly 16x's attention cost |
| Total parameters | Patch projection: `(P^2 * C) * D`; remainder identical to a text Transformer encoder of the same depth and width |
| Active parameters | All parameters active for every patch (dense, absent MoE variants) |
| Persistent inference state | None — ViT is typically used for single-pass classification/feature extraction, not autoregressive generation, so there is no growing cache |
| Communication | Same as a standard Transformer encoder's attention communication pattern |

## Strengths

- Global receptive field from the very first layer — any two patches can interact directly, unlike a CNN's depth-dependent receptive field growth.
- Unifies vision architecture with the Transformer tooling, scaling recipes, and pretraining methodology developed for language.
- Scales well with data and compute — larger ViTs trained on larger datasets have continued to improve without hitting the architectural ceiling CNNs eventually showed.

## Limitations and Failure Modes

- High-resolution images produce large patch counts, and attention's quadratic cost in patch count makes very high resolutions expensive (halving patch size quadruples token count and multiplies attention cost roughly 16x).
- Without a CNN's built-in locality prior, ViT needs more training data (or explicit data augmentation / distillation) to reach comparable performance at moderate dataset sizes — the original ViT paper found it underperformed ResNets when trained on ImageNet-scale data alone, only pulling ahead with much larger pretraining datasets (JFT-300M).
- Fixed patch size discretizes the image; very small or texture-heavy details smaller than one patch are not directly visible to the position where they occur.

## Architecture vs Training Objective

The patchify-and-encode architecture is fixed once patch size and embedding dimension are chosen. Whether the model is trained for supervised classification, contrastive learning (e.g. CLIP-style dual encoders), or self-supervised masked-patch prediction is a training-objective choice layered on top of the same architectural backbone.

## When to Use It

Use ViT when large-scale pretraining data and compute are available and the task benefits from modeling long-range or non-local visual relationships, or when unifying vision and language processing under one shared architecture (multimodal models) is valuable.

## When Not to Use It

Avoid plain ViT in small-data or compute-constrained regimes where a CNN's built-in locality bias gives better sample efficiency, or where very high input resolution makes quadratic patch-count attention cost prohibitive without hierarchical or windowed attention modifications.

## Comparison with Alternatives

- **CNNs** hard-code locality and translation equivariance, giving better sample efficiency at small scale but a harder time modeling long-range dependencies without added depth (see Convolutional Neural Networks).
- **Hierarchical ViTs** (e.g. Swin Transformer) reintroduce a multi-scale, locally-windowed structure to recover some of a CNN's efficiency and inductive bias while keeping attention as the core mechanism.
- **CNN-Transformer hybrids** use a convolutional stem for early, cheap, local feature extraction before switching to global attention for later layers.

## Representative Models

| Model | Patch size | Layers | Hidden dim |
|---|---|---|---|
| ViT-Base (Dosovitskiy et al., 2021) | 16x16 | 12 | 768 |
| ViT-Large (Dosovitskiy et al., 2021) | 16x16 | 24 | 1024 |
| ViT-Huge (Dosovitskiy et al., 2021) | 14x14 | 32 | 1280 |

## References

- Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).
- Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* [arXiv:2103.14030](https://arxiv.org/abs/2103.14030).

[Back to index](../INDEX.md)
