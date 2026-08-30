# AI Model Architectures

A structured field guide to the architectures behind modern AI — from deep-learning foundations through Transformers, Mixture-of-Experts, state-space models, multimodal architectures, test-time memory, JEPA-style predictive models, world models and embodied AI.

```text
FOUNDATIONS → CLASSICAL NETWORKS → ATTENTION/TRANSFORMERS → FOUNDATION MODELS
→ SPARSITY/MoE → SSM & RECURRENT ALTERNATIVES → GENERATIVE → MULTIMODAL
→ WORLD MODELS → MEMORY → REASONING → EMBODIED AI → HYBRIDS → EFFICIENCY
```

## Why This Exists

AI discussions often mix primitives, architecture families, scaling patterns, learning objectives, model families and complete AI systems. This repository separates those layers.

Examples:

- **Attention** is a computational primitive.
- **Transformer** is an architecture family.
- **Mixture-of-Experts** is a conditional-computation/scaling pattern often inserted into a Transformer.
- **JEPA** is a predictive representation architecture/objective family and may use Transformer encoders.
- **V-JEPA** is a reference model family.
- **RAG or multi-agent orchestration** is mainly system architecture, not model architecture.

## How This Repository Is Organized

The numbered folders form the learning path. Each section starts simple, then adds computational structure, trade-offs and representative research.

See:
- [Knowledge Map](00-navigation-and-methodology/knowledge-map.md)
- [Full Index](INDEX.md)
- [Glossary](glossary/terminology.md)
- [Architecture Template](templates/architecture-page-template.md)
- [Sources](SOURCES.md)

## Scope

This repo focuses on **model architecture**. RAG, MCP, vector databases, tool calling and agent orchestration are intentionally kept outside the core taxonomy.

## Status

Expanded release: full navigable structure with formula-level technical depth, worked numeric examples and verified academic citations across all major architecture families and frontier topics.

## License

MIT.
