# Diffusion Models

## Context and Plain-Language Explanation

A diffusion model learns to reverse a gradual noising process. Training corrupts real data with increasing amounts of noise across many timesteps, and trains a network to predict the noise that was added at each step. Sampling starts from pure noise and repeatedly applies the trained denoiser, stepping back toward a clean sample.

## Problem It Tries to Solve

Directly generating complex, high-dimensional continuous data (a realistic image, in one shot) is hard for a single forward pass to learn well. Breaking generation into many small denoising steps turns one hard problem into many easier ones, each of which is just "remove a bit of known-magnitude noise."

## Core Architectural Idea

The forward process adds Gaussian noise to a clean sample x_0 over T timesteps according to a variance schedule β_t:

```
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε,     ε ~ N(0, I)
```

where ᾱ_t = Π_{s=1}^t (1 - β_s) is the cumulative product of (1 - β_s) up to step t. This closed form means any noised x_t can be sampled in one step directly from x_0, without simulating every intermediate step.

Training uses the simplified DDPM objective: sample a random timestep t, generate the noised x_t via the formula above, and train a network ε_θ to predict the noise that was added:

```
L_simple = E_{t, x_0, ε} [ || ε - ε_θ(x_t, t) ||² ]
```

Sampling reverses this: starting from x_T ~ N(0, I) (pure noise), repeatedly predict the noise with ε_θ and step toward a less-noisy x_{t-1}, for t = T down to 1.

**Worked signal-to-noise example.** Take a simple linear β schedule from β_1 = 0.0001 to β_T = 0.02 over T = 1000 steps (the original DDPM schedule). The signal-to-noise ratio at step t is ᾱ_t / (1 - ᾱ_t). A few representative points along this schedule:

| Timestep t | ᾱ_t (approx.) | Signal fraction √ᾱ_t | Noise fraction √(1-ᾱ_t) |
|---|---|---|---|
| 0 (clean) | 1.000 | 1.00 | 0.00 |
| 100 | ≈0.90 | ≈0.95 | ≈0.32 |
| 500 | ≈0.08 | ≈0.28 | ≈0.96 |
| 1000 (pure noise) | ≈0.00004 | ≈0.006 | ≈1.00 |

Even at t=100, one tenth of the way through the schedule, noise already accounts for about a third of the sample's variance. By the halfway point (t=500) the sample is almost entirely noise (noise fraction ≈0.96). This is why sampling has to walk back through many steps concentrated near the noisy end of the schedule — most of the schedule is spent in a regime where the signal is a small fraction of the total.

## Information Flow

```mermaid
flowchart LR
    X0[Clean data x_0] --> NOISE["Forward process: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε"]
    NOISE --> XT[Noised x_t at random timestep t]
    XT --> DEN["Denoiser ε_θ(x_t, t) predicts ε"]
    DEN --> LOSS["Loss = ‖ε - ε_θ(x_t,t)‖²"]
    XT2[x_T ~ N(0,I)] --> STEP["Sampling: repeatedly denoise, T → 0"]
    STEP --> X0HAT[Generated sample]
```

## Components

| Component | Role |
|---|---|
| Forward (noising) process | Fixed, non-learned schedule that progressively corrupts x_0 into pure noise by step T |
| Variance schedule β_t | Controls how much noise is added at each step; shapes the signal-to-noise trajectory |
| Denoiser network ε_θ | Learned network predicting the noise added at a given timestep; the only trained component |
| Timestep conditioning | Mechanism (e.g. sinusoidal embedding) informing the denoiser which noise level it's looking at |
| Sampler | Algorithm that steps from x_T back to x_0 using the trained denoiser; can be a simple ancestral sampler or a faster ODE/SDE solver |

## Computational Characteristics

| Dimension | Notes |
|---|---|
| training parallelism | High — each training step samples one random t per example, so a batch trains in parallel across examples and timesteps |
| sequence scaling | Depends on the denoiser backbone (U-Net or Transformer) applied to spatial/latent tokens, not on a sequence-length axis in the language-model sense |
| total parameters | Set by the denoiser backbone; typically a single network reused across all timesteps via conditioning |
| active parameters | Same as total; no conditional routing |
| persistent inference state | None across samples; within one sample's generation, the running x_t is carried between sampling steps |
| communication | Standard parallelism; no special communication pattern |

## Strengths

Stable training relative to adversarial approaches — the objective is a simple regression loss (predict the noise), not a two-player game. Excellent media generation quality, especially for images, audio, and video. Flexible conditioning: text, class labels, or other signals can be injected into the denoiser at every step.

## Limitations and Failure Modes

Sampling with the original formulation requires many steps (on the order of hundreds to a thousand) to reach acceptable quality, making it much more expensive per sample than a single-pass GAN generator. High-resolution generation multiplies this cost further, since each step processes the full resolution (or a large latent).

## Architecture vs Training Objective

The forward noising process and the denoising objective define the diffusion *framework*; the backbone that implements ε_θ (a convolutional U-Net or a Transformer, see [diffusion-transformers.md](diffusion-transformers.md)) is a separate architectural choice. Two diffusion models with different backbones share the same training objective and sampling procedure but different internal computation.

## When to Use It

High-fidelity media generation (image, audio, video) where sampling cost is acceptable and training stability matters more than sampling speed.

## When Not to Use It

Latency-critical single-pass generation, where the multi-step sampling cost is prohibitive — a GAN or a distilled/few-step diffusion variant may be a better fit.

## Comparison with Alternatives

GANs trade diffusion's training stability for fast single-pass sampling and less predictable training dynamics. Flow matching (see [flows-and-flow-matching.md](flows-and-flow-matching.md)) is a closely related continuous-generation framework with a different training path formulation, often enabling faster sampling with fewer steps.

## Representative Models

DDPM (Denoising Diffusion Probabilistic Models) established the noising/denoising formulation and simplified training objective used here.

## References

- Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

[Back to index](../INDEX.md)
