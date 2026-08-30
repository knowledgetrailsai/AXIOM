# Generative Adversarial Networks

## Context and Plain-Language Explanation

A GAN trains two networks against each other. A generator maps random noise to samples that should look like real data. A discriminator tries to tell real samples from generated ones. Training pushes the generator to fool the discriminator and the discriminator to keep improving at catching it.

## Problem It Tries to Solve

Some generative approaches (like a VAE) require an explicit likelihood or an explicit reconstruction target. A GAN instead uses a learned critic — the discriminator — as the training signal, sidestepping the need to define or compute an explicit density over the data.

## Core Architectural Idea

The generator G maps a noise vector z (sampled from a simple prior) to a generated sample G(z). The discriminator D maps a sample to a scalar in (0, 1), its estimate of the probability that the sample is real. Training is a minimax game:

```
min_G max_D  E_x~data[log D(x)] + E_z~prior[log(1 - D(G(z)))]
```

Read plainly: D wants to output values near 1 for real x and near 0 for generated G(z), so it maximizes the sum of both terms. G only controls the second term, and wants D(G(z)) to be large (so 1 - D(G(z)) is small, making log(1 - D(G(z))) very negative), so it minimizes that same expression. Both networks are trained by alternating gradient steps: update D to get better at distinguishing real from fake, then update G to get better at fooling the current D. In practice, the naive form of the generator's loss saturates early in training (when D easily rejects G's samples, the gradient signal to G vanishes), so most implementations use a non-saturating variant that trains G to directly maximize log D(G(z)) instead of minimizing log(1 - D(G(z))), which provides a stronger gradient while pursuing the same adversarial goal.

At the theoretical optimum of this game — with both networks having unlimited capacity — the generator's distribution matches the real data distribution exactly, and the discriminator can do no better than random guessing (D(x) = 0.5 everywhere). In practice, finite capacity and optimization dynamics mean this equilibrium is approached, not exactly reached.

## Information Flow

```mermaid
flowchart LR
    Z[Noise z] --> G[Generator G]
    G --> FAKE[Generated sample G(z)]
    DATA[Real sample x] --> D[Discriminator D]
    FAKE --> D
    D --> SCORE["D(x) or D(G(z)): probability of real"]
    SCORE --> LOSSG[Generator loss: fool D]
    SCORE --> LOSSD[Discriminator loss: catch fakes]
```

## Components

| Component | Role |
|---|---|
| Generator G | Maps noise z to a candidate sample; the network that eventually does the generating |
| Discriminator D | Scores samples as real or fake; provides the training signal to G |
| Noise prior | Simple distribution (e.g. Gaussian) that z is sampled from as G's input |
| Adversarial loss | The minimax objective driving both networks' updates |

## Computational Characteristics

| Dimension | Notes |
|---|---|
| training parallelism | Standard, but requires alternating updates between G and D, and their relative update frequency/balance is a training-stability lever |
| sequence scaling | Not inherently sequential; typically applied to fixed-size outputs (images) rather than variable-length sequences |
| total parameters | Sum of generator and discriminator parameters; only the generator is needed at inference/sampling time |
| active parameters | Same as total for each network; no conditional routing |
| persistent inference state | None — a single forward pass through G produces a sample |
| communication | Standard parallelism; no special communication pattern |

## Strengths

Sharp, high-fidelity samples relative to some likelihood-based alternatives at comparable model size. Fast sampling once trained: a single forward pass through G, no iterative refinement needed (unlike diffusion's many denoising steps).

## Limitations and Failure Modes

Training instability: the adversarial game can oscillate rather than converge, especially if G and D are not kept in a reasonable balance of capability. Mode collapse: G can learn to produce a narrow range of outputs that reliably fool the current D, rather than covering the full diversity of the real data distribution. Discriminator and generator capacity and learning rates need careful tuning; a discriminator that gets too strong too fast can leave the generator with no useful gradient.

## Architecture vs Training Objective

The generator and discriminator networks are architecture. The adversarial minimax objective is the training-time mechanism that shapes what they learn — the same generator network trained with a different objective (e.g. as a VAE decoder, or as a diffusion denoiser) is not doing adversarial training at all, even with an identical forward-pass structure.

## When to Use It

Applications prioritizing fast, single-pass sampling and sharp output quality, where the training instability trade-off is acceptable and enough tuning effort is available to keep the adversarial game balanced.

## When Not to Use It

Applications needing stable, reproducible training with minimal hyperparameter sensitivity, or needing an explicit likelihood — diffusion models (see [diffusion-models.md](diffusion-models.md)) trade GANs' fast single-pass sampling for a much more stable training procedure.

## Comparison with Alternatives

Diffusion models generally trade GANs' single-pass, potentially unstable adversarial training for iterative, more stable denoising training at the cost of many sampling steps. VAEs provide an explicit, tractable training objective (the ELBO) but tend to produce blurrier samples than a well-trained GAN.

## Representative Models

Goodfellow et al.'s original GAN formulation is the reference for the minimax objective above; many later variants (conditional GANs, architectures adding progressive growing or style-based generators) build on this same adversarial core.

## References

- Goodfellow, I. et al. (2014). *Generative Adversarial Networks.* [arXiv:1406.2661](https://arxiv.org/abs/1406.2661).

[Back to index](../INDEX.md)
