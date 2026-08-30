# Diffusion vs Autoregressive

## Short Answer

An autoregressive model generates output one unit at a time, each conditioned on everything generated so far, in a strict causal order. A diffusion model generates all positions together, refining a noisy full-size sample over a fixed number of denoising steps. Autoregressive generation cost scales with output length; diffusion generation cost scales with the number of denoising steps, largely independent of output size per step.

## Comparison

| Dimension | Diffusion | Autoregressive |
|---|---|---|
| generation process | iterative denoising of a full sample, all positions updated together at each step | ordered next-unit prediction, one position at a time |
| parallelism per step | all positions can be updated in parallel within a single denoising step | causal order is inherently sequential — position t needs position t-1 already generated |
| number of model calls | one call per denoising step, typically tens (e.g. 20-50 steps for common image samplers) regardless of output resolution | one call per output unit — a 1,000-token generation needs roughly 1,000 sequential calls (before speculative decoding or batching tricks) |
| classic strength | continuous media: images, audio, video | discrete sequences: text, code |
| backbone | U-Net (classic) or Transformer (modern, e.g. DiT) | causal (decoder-only) Transformer |
| exact causal structure | none required — the whole sample is denoised jointly | required by construction — training uses a causal mask precisely to enforce it |

## The Real Trade-off

Generating a 1,024-token document autoregressively takes on the order of 1,024 sequential forward passes (absent parallel-decoding tricks), each one dependent on the previous token actually being generated. Generating an image with a diffusion model at any resolution typically takes a fixed, resolution-independent number of denoising steps — commonly 20-50 for a standard sampler — regardless of whether the image is small or large, because every step processes the whole image at once. The cost axes are different by construction: autoregressive cost scales with *output length*; diffusion cost scales with *number of denoising steps*, a hyperparameter largely decoupled from output size.

This is also why diffusion suits continuous media well: there is no natural "causal order" over pixels the way there is a natural left-to-right order over text, so an objective that denoises the whole sample jointly avoids inventing an arbitrary generation order.

## Hybrid Possibilities

The boundaries are blurring in both directions: autoregressive image models exist (generating images as a sequence of discrete visual tokens), and diffusion language models exist (denoising a full sequence of text tokens jointly instead of generating left-to-right). Architecture and generation objective are increasingly separable choices rather than a fixed pairing.

## References

- Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
- Peebles, W. & Xie, S. (2023). *Scalable Diffusion Models with Transformers.* [arXiv:2212.09748](https://arxiv.org/abs/2212.09748).

[Back to index](../INDEX.md)
