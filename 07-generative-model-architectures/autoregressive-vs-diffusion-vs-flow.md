# Autoregressive vs Diffusion vs Flow

## Short Answer

Autoregressive models generate one unit at a time, conditioned on everything generated so far, and compute exact likelihoods naturally. Diffusion models generate by iterative denoising, need many model calls per sample, and compute only an approximate (variational lower bound) likelihood. Flow / flow-matching models transport noise to data along a continuous path, need fewer model calls than typical diffusion, and (for classical invertible flows specifically) can give exact likelihoods; flow-matching's ODE-based sampling generally gives approximate likelihood in practice. None of the three is tied to one specific network backbone — a Transformer can implement the autoregressive predictor, the diffusion denoiser, or the flow's vector field.

## Comparison

| Property | Autoregressive | Diffusion | Flow / Flow Matching |
|---|---|---|---|
| generation unit | next discrete/continuous unit, conditioned on all previous units | iterative denoising across noise levels | continuous transport along a learned vector field |
| per-step parallelism | low — output order is causal, later units depend on earlier ones | high — computation at each denoising step is parallel across all positions | high — computation at each solver step is parallel across all positions |
| typical domains | text, discrete sequences, tokenized media | images, audio, video | continuous data or latents |
| model calls per sample | roughly proportional to output length (one call per generated unit, or per block with speculative decoding) | roughly proportional to number of denoising steps — commonly cited as needing on the order of hundreds to ~1,000 steps for original DDPM sampling, though modern fast samplers or distilled models often use roughly 20–50 (treat both figures as commonly cited approximate ranges, not precise universal counts) | roughly proportional to ODE solver steps, often fewer than typical diffusion sampling because flow-matching's straight-line training paths are easier to integrate accurately with a coarse solver |
| likelihood | exact, via the chain rule of probability over the generation order | approximate — trained on a variational bound (the ELBO-derived simplified loss), not an exact density | exact for classical invertible normalizing flows (tractable Jacobian by construction); flow-matching's ODE-based sampling generally gives only approximate likelihood in practice |
| training stability | generally stable — a straightforward next-unit prediction loss | generally stable — a straightforward denoising regression loss | generally stable — flow matching uses a straightforward regression loss; classical flows are stable but architecturally constrained by invertibility |
| common backbone | causal Transformer | U-Net or Transformer (DiT) | neural network parameterizing a vector field, often a Transformer or U-Net |

## The Real Trade-off

The three families trade off *when* computation happens relative to *how much* of the output is fixed at each step. Autoregressive generation commits to one unit at a time and can never revise an earlier choice, which gives exact likelihood but forces strictly sequential, low-parallelism generation. Diffusion and flow-matching instead refine the *entire* output jointly across many steps, so every step is fully parallel across positions, at the cost of needing multiple full passes through the network to produce one sample. Architecture (the backbone) is a separate axis entirely from this trade-off — a Transformer backbone appears in all three families, and the choice of family is really a choice about the *generation and training procedure*, not about which network architecture is allowed.

## Hybrid Possibilities

Some systems combine an autoregressive Transformer over discrete tokens (e.g. VQ-VAE codebook indices, see [autoencoders-vae-vqvae.md](autoencoders-vae-vqvae.md)) with a diffusion or flow-based decoder for the final continuous output, using each family where its trade-off is most favorable — autoregressive for a compact discrete sequence, diffusion or flow for high-fidelity continuous rendering.

## References

- Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
- Lipman, Y. et al. (2023). *Flow Matching for Generative Modeling.* [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- Peebles, W. & Xie, S. (2023). *Scalable Diffusion Models with Transformers.* [arXiv:2212.09748](https://arxiv.org/abs/2212.09748).

[Back to index](../INDEX.md)
