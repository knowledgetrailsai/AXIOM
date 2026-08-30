# Autoregressive Language Models

## Context and Plain-Language Explanation

An autoregressive model predicts the next unit of a sequence given everything before it, then feeds its own output back in as input to predict the next one. This turns the hard problem of modeling a joint probability over an entire sequence into a sequence of much simpler next-step predictions.

## Why This Architecture Exists

In practical terms, **Autoregressive Language Models** is useful because it addresses a limitation that simpler approaches face. The next paragraph explains that limitation in technical detail; first, keep in mind the real-world goal: making the model more useful, efficient, reliable, or capable for a particular kind of task.

Modeling the joint probability `P(x_1, ..., x_n)` directly over an entire sequence is intractable for any realistic vocabulary and length — the number of possible sequences grows exponentially. Autoregressive factorization turns this into `n` tractable conditional predictions instead.

## Core Architectural Idea

The chain rule of probability factors any joint distribution exactly, without approximation:

`P(x_1, ..., x_n) = Π_{i=1}^{n} P(x_i | x_1, ..., x_{i-1})`

A causal Transformer decoder (see Encoder, Decoder and Encoder-Decoder Transformers) parameterizes each conditional `P(x_i | x_<i)` with a shared set of weights, so the same network predicts every position's distribution given only the prefix causal attention allows it to see.

### Cross-entropy training loss

Training minimizes the negative log-likelihood the factorization assigns to the true sequence — the cross-entropy between the predicted distribution and the true next token:

`L = - Σ_i log P(x_i | x_<i))`

**Worked example.** A 4-token vocabulary `{A, B, C, D}`, true next token is `C`. Suppose the model's softmax output is:

```
P = [P(A)=0.10, P(B)=0.20, P(C)=0.55, P(D)=0.15]
```

Cross-entropy loss for this single step:

```
L = -log(P(C)) = -log(0.55) = 0.5978 nats
```

If instead the model had been confident and correct, say `P(C) = 0.95`:

```
L = -log(0.95) = 0.0513 nats
```

And if it had been confidently wrong, `P(C) = 0.02`:

```
L = -log(0.02) = 3.912 nats
```

Cross-entropy penalizes confident wrong predictions far more heavily than uncertain ones — the loss grows without bound as the assigned probability to the true token approaches zero, which is what drives the model to avoid overconfident mistakes during training.

## Information Flow

```mermaid
flowchart LR
    Prefix[x_1 ... x_i-1] --> Model[Causal Transformer]
    Model --> Dist[P(x_i given prefix)]
    Dist --> Sample[Sample or take argmax: x_i]
    Sample --> Prefix
```

## Components

| Component | Role |
|---|---|
| Causal attention mask | Restricts each position to only its own prefix, matching train-time factorization to inference-time generation |
| Shared weights across positions | The same conditional distribution function `P(x_i | x_<i)` is applied at every position |
| Output head | Projects the final hidden state at position `i` to a distribution over the vocabulary |
| Sampling strategy (inference only) | Greedy, temperature, top-k, or top-p sampling turns the distribution into an actual next token |

## Computational Characteristics

| Dimension | Notes |
|---|---|
| Training parallelism | Fully parallel across all positions during training — the entire target sequence is known, so all conditionals can be computed in one forward pass under the causal mask |
| Sequence scaling | Training: `O(n^2)` from the underlying causal attention; inference: strictly sequential, one token at a time, `O(n)` total steps each reusing the growing KV cache |
| Total parameters | Same as the underlying Transformer decoder — this is an objective/factorization choice, not an added component |
| Active parameters | Same as the underlying architecture (dense unless combined with MoE) |
| Persistent inference state | Growing KV cache during generation, since each new token attends to every previous one |
| Communication | Same as the underlying causal Transformer's attention communication pattern |

## Strengths

- Exact, lossless factorization of the joint sequence probability — no approximation is introduced by the chain rule itself.
- Training is fully parallel despite the model being sequential at inference, since teacher forcing supplies the true prefix at every position simultaneously.
- Naturally supports open-ended generation of arbitrary length, unlike architectures that require a fixed output size.

## Limitations and Failure Modes

- Inference is inherently sequential — generating `n` tokens requires `n` forward passes (amortized by the KV cache, but still `n` sequential steps).
- Maximizing likelihood does not directly optimize for truthfulness, helpfulness, or successful multi-step planning — a well-calibrated next-token predictor can still generate confidently incorrect continuations if the training distribution supports them.
- Errors can compound during generation: an early wrong token becomes part of the "true" prefix used for the next prediction, a mismatch between train-time teacher forcing and inference-time self-conditioning known as exposure bias.

## Architecture vs Training Objective

Autoregressive factorization is a training-objective and generation-protocol choice, not itself a specific network architecture — it can be applied with an RNN, a causal Transformer, or in principle any causal sequence model. The causal Transformer decoder is simply the dominant current architecture used to parameterize it (see Encoder, Decoder and Encoder-Decoder Transformers).

## When to Use It

Use autoregressive modeling for open-ended sequence generation of arbitrary length where left-to-right (or otherwise fixed-order) causal factorization matches the task — general text generation, code generation, and any setting where output length is not known in advance.

## When Not to Use It

Avoid pure autoregressive factorization when bidirectional context is essential for the actual task and no left-to-right generation is required (representation learning, classification) — masked/denoising objectives (see Masked and Denoising Language Models) fit those cases better. Consider non-autoregressive or parallel-decoding methods when generation latency, not quality, is the binding constraint.

## Comparison with Alternatives

- **Masked/denoising objectives** predict corrupted spans using full bidirectional context rather than strictly left-to-right conditionals, at the cost of not being a valid generative sampling procedure for arbitrary-length open-ended text.
- **Diffusion/non-autoregressive generation** predicts many positions in fewer sequential steps by iterative refinement, trading exact chain-rule factorization for parallel-decoding speed.

## Representative Models

| Model | Unit predicted | Notable property |
|---|---|---|
| GPT-2 / GPT-3 | Subword tokens | Canonical decoder-only autoregressive LLM |
| PixelRNN / PixelCNN | Image pixels | Chain-rule factorization applied to images |
| Byte Latent Transformer | Bytes (dynamically patched) | Autoregressive modeling without a fixed tokenizer |

## References

- Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model.* Journal of Machine Learning Research, 3, 1137-1155.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2).* OpenAI.
- Pagnoni, A. et al. (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens.* [arXiv:2412.09871](https://arxiv.org/abs/2412.09871).

[Back to index](../INDEX.md)
