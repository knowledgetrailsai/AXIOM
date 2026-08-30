# LLaMA-Style Dense Decoders

LLaMA is a useful public reference for a modern **dense causal decoder** design: a decoder-only Transformer (see Transformer Families, Autoregressive Language Models) using a specific, publicly documented combination of choices:

- RoPE for position information (see Position Encoding and RoPE).
- RMSNorm, pre-norm placement (see Normalization and Residual Connections).
- SwiGLU-gated FFN instead of a plain two-layer FFN (see Transformer Block).
- Grouped-Query Attention in later versions, reducing KV-cache memory (see MQA, GQA and KV Cache).

Treat "LLaMA-style" as a reference family for these specific, documented architectural choices — not as the definition of what a modern LLM must be. Other dense decoder families (Mistral, Qwen, Gemma) make similar but not identical choices.

## Representative models

| Model | Layers | Hidden dim | Attention heads | KV heads | Parameters |
|---|---|---|---|---|---|
| LLaMA 7B (Touvron et al., 2023) | 32 | 4096 | 32 | 32 (MHA) | 6.7B |
| LLaMA 13B (Touvron et al., 2023) | 40 | 5120 | 40 | 40 (MHA) | 13.0B |
| LLaMA 2 7B (Touvron et al., 2023) | 32 | 4096 | 32 | 32 (MHA) | 6.7B |
| LLaMA 2 70B (Touvron et al., 2023) | 80 | 8192 | 64 | 8 (GQA) | 68.9B |

LLaMA 2 70B's shift to 8 KV heads instead of 64 illustrates the GQA memory-reduction argument directly: applying the KV cache formula from MQA, GQA and KV Cache, this is an 8x reduction in KV-cache memory relative to full MHA at the same model size.

## References

- Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* [arXiv:2302.13971](https://arxiv.org/abs/2302.13971).
- Touvron, H. et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models.* [arXiv:2307.09288](https://arxiv.org/abs/2307.09288).

[Back to index](../INDEX.md)
