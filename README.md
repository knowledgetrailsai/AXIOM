# Axiom

A structured field guide to the architectures behind modern AI — spanning deep-learning foundations through Transformers, Mixture-of-Experts, state-space models, multimodal architectures, test-time memory, JEPA-style predictive models, world models and embodied AI.

```text
FOUNDATIONS → CLASSICAL NETWORKS → ATTENTION/TRANSFORMERS → FOUNDATION MODELS
→ SPARSITY/MoE → SSM & RECURRENT ALTERNATIVES → GENERATIVE → MULTIMODAL
→ WORLD MODELS → MEMORY → REASONING → EMBODIED AI → HYBRIDS → EFFICIENCY
```

## Table of Contents

- [Why This Exists](#why-this-exists)
- [How This Repository Is Organized](#how-this-repository-is-organized)
  - [Knowledge Map](00-navigation-and-methodology/knowledge-map.md)
  - [Full Index](INDEX.md)
  - [Glossary](glossary/terminology.md)
  - [Architecture Template](templates/architecture-page-template.md)
  - [Sources](SOURCES.md)
- [Learning Path](#learning-path)
  - [00 - Navigation and Methodology](00-navigation-and-methodology/README.md)
  - [01 - Deep-Learning Foundations](01-deep-learning-foundations/README.md)
  - [02 - Classical Neural Architectures](02-classical-neural-architectures/README.md)
  - [03 - Attention and Transformers](03-attention-and-transformers/README.md)
  - [04 - Foundation Model Architectures](04-foundation-model-architectures/README.md)
  - [05 - Sparse and Mixture-of-Experts](05-sparse-and-mixture-of-experts/README.md)
  - [06 - State-Space and Recurrent Alternatives](06-state-space-and-recurrent-alternatives/README.md)
  - [07 - Generative Model Architectures](07-generative-model-architectures/README.md)
  - [08 - Multimodal Architectures](08-multimodal-architectures/README.md)
  - [09 - Predictive and World Models](09-predictive-and-world-models/README.md)
  - [10 - Memory and Adaptive Computation](10-memory-and-adaptive-computation/README.md)
  - [11 - Reasoning-Oriented Architectures](11-reasoning-oriented-architectures/README.md)
  - [12 - Embodied and Robotics Models](12-embodied-and-robotics-models/README.md)
  - [13 - Hybrid Architectures](13-hybrid-architectures/README.md)
  - [14 - Model Efficiency and Inference](14-model-efficiency-and-inference/README.md)
  - [15 - Architecture Comparisons](15-architecture-comparisons/README.md)
  - [16 - Architecture Patterns](16-architecture-patterns/README.md)
  - [17 - Reference Model Families](17-reference-model-families/README.md)
  - [18 - Research Frontier](18-research-frontier/README.md)
- [Scope](#scope)
- [Status](#status)
- [License](#license)

## Why This Exists

AI discussions often mix primitives, architecture families, scaling patterns, learning objectives, model families and complete AI systems — this repository separates those layers.

Examples:

- **Attention** is a computational primitive.
- **Transformer** is an architecture family.
- **Mixture-of-Experts** is a conditional-computation/scaling pattern often inserted into a Transformer.
- **JEPA** is a predictive representation architecture/objective family and may use Transformer encoders.
- **V-JEPA** is a reference model family.
- **RAG or multi-agent orchestration** is mainly system architecture, not model architecture.

## How This Repository Is Organized

### Learning Path

The numbered folders form the learning path — each section starts simple, then layers on computational structure, trade-offs and representative research.

### How to Read a Page

Each architecture page answers the same practical questions:

1. **What is it?** A short explanation before the formal terminology.
2. **What problem does it solve?** The limitation or cost that motivated the design.
3. **How does it work?** The data flow, equations and components.
4. **What does it cost?** Training and inference compute, memory, latency and hardware considerations.
5. **When is it useful?** Concrete application scenarios and deployment constraints.
6. **What should it be compared with?** Alternatives that solve a similar problem, and the trade-offs between them.

Commercial model names are included as examples of where an architectural pattern has appeared in a public model or product. They are not proof that a company uses one exact implementation internally — proprietary details are often unavailable. The architecture, training objective, data, post-training and serving system should be evaluated separately.

See:
- [Knowledge Map](00-navigation-and-methodology/knowledge-map.md)
- [Full Index](INDEX.md)
- [Glossary](glossary/terminology.md)
- [Architecture Template](templates/architecture-page-template.md)
- [Sources](SOURCES.md)

## Scope

This repo focuses on **model architecture** — RAG, MCP, vector databases, tool calling and agent orchestration are intentionally kept outside the core taxonomy.

## Relationship to companion repositories

Axiom underpins [OASIS](https://github.com/knowledgetrailsai/OASIS) Chapter 14's model-selection guidance, but unlike Forge, Loom, Helm, Verity, Compass, and Fulcrum, it is not itself a Part III chapter companion — see the [Companion Repository Index](https://github.com/knowledgetrailsai/OASIS/blob/main/References/companion-repository-index.md) for the full map.

- **[Forge](https://github.com/knowledgetrailsai/Forge)** (Chapter 15, data and knowledge engineering): its [long-context-vs-rag.md](https://github.com/knowledgetrailsai/Forge/blob/main/07-advanced-retrieval-architectures/long-context-vs-rag.md) and [embedding-model-selection.md](https://github.com/knowledgetrailsai/Forge/blob/main/08-embeddings-and-indexing/embedding-model-selection.md) make retrieval-architecture tradeoffs that depend on the attention-mechanism and embedding background covered in this repo's [long-context-and-efficient-attention.md](03-attention-and-transformers/long-context-and-efficient-attention.md) and [embeddings.md](01-deep-learning-foundations/embeddings.md) — Axiom owns the underlying mechanism, Forge owns the applied retrieval decision.
- **[Ageis](https://github.com/knowledgetrailsai/Ageis)** uses "model" to mean a coding-tool deployment model, not model architecture; there is no direct content dependency between the two repos despite the shared word.

## Status

Expanded release: full navigable structure with formula-level technical depth, worked numeric examples and verified academic citations across all major architecture families and frontier topics.

## License

Licensed under [CC BY-SA 4.0](https://github.com/knowledgetrailsai/OASIS/blob/main/LICENSE.md). Reuse and adaptation are welcome with credit to KnowledgeTrails-OASIS, a link to the license, an indication of changes, and release of adaptations under the same license.

## About Us

**Shripadraj Mujumdar** is an Agentic AI & Automation Strategist, Advisor, and Responsible AI Expert with 28+ years of experience in enterprise architecture and AI-driven transformation, including deep hands-on work in Agentic AI, Generative AI, and enterprise data and knowledge platforms. His practice spans designing multi-agent systems, knowledge-graph and RAG architectures, accelerated delivery capabilities, and Responsible AI governance frameworks aligned to global regulatory standards. This methodology ecosystem distills that practitioner experience — architecture, delivery, evaluation, governance, and economics — into a single, reusable body of work.

**Ankit Mirajkar** is a Data & AI Architect and technology consultant specializing in modern data platforms, enterprise data architecture, and Agentic AI. His expertise spans scalable data engineering, AI-ready data platforms, Generative AI, and cloud technologies, with a strong focus on turning complex data challenges into practical, production-ready solutions. He also works at the intersection of architecture, technology strategy, and innovation to help organizations build intelligent, scalable data ecosystems.
