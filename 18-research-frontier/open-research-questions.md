# Open Research Questions

Each question below is stated at the level of a specific, checkable claim, not a general worry. None of these are settled by public research as of this writing.

## Sequence Modeling

Whether decoder-only autoregressive scaling laws — the compute/data/parameter trade-off curves established for dense Transformers — hold in the same form for sparse MoE architectures and SSM-hybrid architectures at matched total-compute budgets is not yet settled by public research as of this writing. Related and more specific: can a fixed-size compressed state (SSM-style) match attention's exact-retrieval accuracy on tasks requiring precise recall of a specific earlier token, while keeping the same streaming, constant-memory efficiency?

## Memory

Can a neural memory module that updates its own weights at test time (Titans-style, see [titans-test-time-memory.md](../10-memory-and-adaptive-computation/titans-test-time-memory.md)) run over arbitrarily long sessions without drift, cross-user information leakage, or catastrophic overwriting of earlier important content — and if so, under what bound on session length or update frequency?

## Sparsity

At what combination of batch size, expert count, and interconnect bandwidth does routed MoE compute (see [dense-vs-moe.md](../15-architecture-comparisons/dense-vs-moe.md)) reliably beat dense compute of matched active-parameter cost, once all-to-all communication overhead is included — and does that crossover point hold consistently across hardware generations?

## Tokenization

Will dynamic byte/patch architectures (see [tokenizer-free-and-byte-level-models.md](tokenizer-free-and-byte-level-models.md)) match fixed-subword-vocabulary models' training and inference efficiency at frontier model scale, or does the dynamic-patching mechanism itself introduce a compute or optimization cost that offsets its fragmentation benefits at scale?

## Reasoning

For a fixed inference-compute budget, does allocating extra computation to generated chain-of-thought tokens, latent recurrent depth (see [latent-reasoning.md](../11-reasoning-oriented-architectures/latent-reasoning.md)), explicit search with a verifier (see [generator-verifier.md](../11-reasoning-oriented-architectures/generator-verifier.md)), or world-model rollouts produce the best accuracy per unit of compute, and does the answer depend on problem class in a way that is predictable in advance?

## World Models

Does a world model that achieves low prediction error on held-out passive observation data actually learn dynamics that generalize correctly to action sequences never seen during training, or only correlational structure specific to the training distribution's action patterns (see [world-model-research.md](world-model-research.md))?

## Embodiment

How much robot-specific interaction data is required for a vision-language-action model (see [vision-language-action-models.md](../12-embodied-and-robotics-models/vision-language-action-models.md)) fine-tuned from a web-scale VLM to reliably transfer its pretrained visual-semantic knowledge into physically safe, precise control, across embodiments the fine-tuning data did not cover?

## Evaluation

Given a benchmark score improvement, is there a reliable methodology to attribute how much of that improvement came from architecture change versus data scale, training objective, post-training, or added test-time compute — and in the absence of that methodology, how much can any single architecture comparison in this repository actually be trusted to isolate architecture as the causal factor?

[Back to index](../INDEX.md)
