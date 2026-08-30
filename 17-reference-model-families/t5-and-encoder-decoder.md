# T5 and Encoder-Decoder Models

T5 is the reference for Transformer encoder-decoder design (see Encoder, Decoder and Encoder-Decoder Transformers):

- Bidirectional source encoder — full self-attention, no causal mask.
- Causal target decoder — masked self-attention, standard autoregressive generation.
- Cross-attention linking the two: decoder queries read encoder keys/values directly.
- Denoising span-corruption pretraining: contiguous spans of the input are replaced with sentinel tokens, and the decoder is trained to generate the removed spans as a target sequence (see Masked and Denoising Language Models for the worked example).

## Unified text-to-text framing

T5's other architectural lesson is treating every task — classification, translation, summarization, regression — as text-to-text: the input and output are both plain text, with the task itself specified as part of the input text (a task prefix). This does not change the encoder-decoder architecture; it is a data-formatting convention that lets one architecture and one training objective cover many task types.

## Representative models

| Model | Layers (enc/dec) | Hidden dim | Parameters |
|---|---|---|---|
| T5-Small (Raffel et al., 2020) | 6 / 6 | 512 | 60M |
| T5-Base (Raffel et al., 2020) | 12 / 12 | 768 | 220M |
| T5-Large (Raffel et al., 2020) | 24 / 24 | 1024 | 770M |
| T5-11B (Raffel et al., 2020) | 24 / 24 | 1024 | 11B |

## References

- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* [arXiv:1910.10683](https://arxiv.org/abs/1910.10683).

[Back to index](../INDEX.md)
