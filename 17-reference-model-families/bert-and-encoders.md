# BERT and Encoder Models

BERT is the reference for **encoder-only Transformer** architecture combined with masked-language-model pretraining (see Transformer Families and Masked and Denoising Language Models).

## Architectural lesson

- Bidirectional self-attention with no causal mask — every position sees every other position, including "future" ones.
- Contextual token representations: the same word gets a different vector depending on surrounding context, unlike a static embedding lookup.
- No causal generation requirement — BERT produces representations for classification, tagging, and retrieval, not open-ended text.
- Trained with the 80/10/10 masked-language-model rule: of the 15% of tokens selected for the objective, 80% become `[MASK]`, 10% become a random token, 10% stay unchanged, with the loss always computed against the true original token (see Masked and Denoising Language Models for the worked numeric example).

## Representative models

| Model | Layers | Hidden dim | Attention heads | Parameters |
|---|---|---|---|---|
| BERT-Base (Devlin et al., 2019) | 12 | 768 | 12 | 110M |
| BERT-Large (Devlin et al., 2019) | 24 | 1024 | 16 | 340M |
| RoBERTa-Base (Liu et al., 2019) | 12 | 768 | 12 | 125M |
| RoBERTa-Large (Liu et al., 2019) | 24 | 1024 | 16 | 355M |

## References

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* [arXiv:1907.11692](https://arxiv.org/abs/1907.11692).

[Back to index](../INDEX.md)
