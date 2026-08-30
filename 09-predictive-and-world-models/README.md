# Predictive and World Models

This section covers the architecture concepts grouped under **Predictive and World Models**.

## Core Idea

This chapter introduces the central ideas behind predictive and world models. Read the linked pages as a progression from the problem being solved, to the mechanism that solves it, to the trade-offs that appear in real systems. The goal is to understand not only what a model is called, but what computation it performs and why that computation is useful.

## How to Use This Chapter

For each architecture, ask: What problem does it solve? What information or state does it preserve? How does computation scale with input size? What happens to memory, latency, and hardware cost? Which capabilities come from the architecture, and which come from data, training, post-training, or the surrounding product system?

## Practical Applicability

Use the chapter to form a design hypothesis, then test it on representative data and hardware. Compare quality, throughput, latency, peak memory, reliability, and operating cost—not just parameter count or theoretical FLOPs. The linked pages identify publicly documented research and commercial model examples where available; proprietary model names do not reveal every implementation detail.

## Contents

- [What Is A World Model](what-is-a-world-model.md)
- [Latent World Models](latent-world-models.md)
- [Jepa](jepa.md)
- [I Jepa And V Jepa](i-jepa-and-v-jepa.md)
- [V Jepa 2](v-jepa-2.md)
- [Generative World Models And Genie](generative-world-models-and-genie.md)
- [Planning With World Models](planning-with-world-models.md)
- [Predictive Vs Generative World Models](predictive-vs-generative-world-models.md)

## Questions to Ask

- What problem does the architecture solve?
- What computation or state mechanism is new?
- How does information move through the model?
- How does training and inference scale?
- What are the memory and hardware implications?
- Which capabilities are architectural, and which come from data, training or test-time methods?

[Back to repository index](../INDEX.md)
