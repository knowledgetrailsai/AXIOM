# Foundation Model Architectures

This section covers the architecture concepts grouped under **Foundation Model Architectures**.

## Core Idea

This chapter introduces the central ideas behind foundation model architectures. Read the linked pages as a progression from the problem being solved, to the mechanism that solves it, to the trade-offs that appear in real systems. The goal is to understand not only what a model is called, but what computation it performs and why that computation is useful.

## How to Use This Chapter

For each architecture, ask: What problem does it solve? What information or state does it preserve? How does computation scale with input size? What happens to memory, latency, and hardware cost? Which capabilities come from the architecture, and which come from data, training, post-training, or the surrounding product system?

## Practical Applicability

Use the chapter to form a design hypothesis, then test it on representative data and hardware. Compare quality, throughput, latency, peak memory, reliability, and operating cost—not just parameter count or theoretical FLOPs. The linked pages identify publicly documented research and commercial model examples where available; proprietary model names do not reveal every implementation detail.

## Contents

- [Autoregressive Language Models](autoregressive-language-models.md)
- [Masked And Denoising Language Models](masked-and-denoising-language-models.md)
- [Vision Transformers](vision-transformers.md)
- [Foundation Model Design Patterns](foundation-model-design-patterns.md)

## Questions to Ask

- What problem does the architecture solve?
- What computation or state mechanism is new?
- How does information move through the model?
- How does training and inference scale?
- What are the memory and hardware implications?
- Which capabilities are architectural, and which come from data, training or test-time methods?

[Back to repository index](../INDEX.md)
