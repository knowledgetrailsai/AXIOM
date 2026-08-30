# GPT-Style Decoder Models

The broad pattern is a **decoder-only causal Transformer**:

`tokens → causal attention + FFN blocks → output distribution`

trained with the autoregressive next-token objective (see Autoregressive Language Models and Transformer Families).

## Architectural lesson

Named model families differ in specific choices layered on top of this shared pattern: position encoding (learned absolute vs. RoPE vs. ALiBi), attention grouping (MHA vs. GQA vs. MQA), normalization variant and placement (LayerNorm vs. RMSNorm, pre-norm vs. post-norm), activation function (GELU vs. SwiGLU), context length, and post-training method. None of these choices change the core decoder-only causal pattern itself.

Do not infer proprietary architecture details (exact layer counts, attention configuration, training data) for closed models beyond what is officially documented — treat undocumented specifics as unknown, not as inferable from behavior or benchmark scores.

## Representative models (publicly documented architecture details only)

| Model | Layers | Hidden dim | Attention heads | Parameters |
|---|---|---|---|---|
| GPT-2 (Radford et al., 2019, largest) | 48 | 1600 | 25 | 1.5B |
| GPT-3 (Brown et al., 2020, largest) | 96 | 12288 | 96 | 175B |

## References

- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners (GPT-3).* [arXiv:2005.14165](https://arxiv.org/abs/2005.14165).

[Back to index](../INDEX.md)
