# Master Comparison Matrix

| Architecture | Core idea | Sequence behavior | Inference memory | Strong fit | Primary trade-off |
|---|---|---|---|---|---|
| MLP | dense transformation | n/a | none | generic representation | no structure |
| CNN | local shared filters | grid-local | none | vision/signals | weak global interaction |
| RNN/LSTM | recurrent state | linear recurrent | compact | streaming | sequential training |
| Transformer | attention | explicit pairwise context | growing KV cache | foundation models | long-context cost |
| MoE Transformer | routed experts | attention + sparse FFN | KV cache + all weights resident | huge capacity at fixed active compute | communication and routing overhead |
| Mamba/SSM | selective state | linear | compact | long/streaming sequences | compressed history, weaker exact recall |
| RWKV/xLSTM | modern recurrence | linear recurrent | compact | streaming LM | less mature tooling/ecosystem |
| Diffusion | denoising | iterative | current latent only | media generation | many sampling steps per output |
| Flow | vector field | solver-based | current latent only | continuous generation | numerical integration cost |
| Dual encoder | aligned embeddings | encoder-dependent | embeddings only | retrieval | limited cross-modal fusion depth |
| JEPA | latent target prediction | encoder-dependent | latent state | world representation learning | low inspectability |
| Generative world model | world simulation | temporal rollout, one frame per step | full world state per frame | interactive simulation | realism does not imply causal correctness |
| VLA | multimodal input → action tokens | autoregressive over action tokens | context/state (KV cache) | robotics with language instructions | safety and control-frequency limits |
| Titans-like | adaptive neural memory | hybrid attention + test-time-updated memory | persistent, mutable memory state | very long effective history | test-time weight updates complicate serving |

## How to actually use this table

Read this table by starting from a concrete requirement, not from an architecture name. Pick the row whose "strong fit" column matches your task, then check its "primary trade-off" column to see what you are giving up.

**Worked example 1: "I need to process a 2-hour audio transcript and answer questions about any part of it, on a tight cost budget."** Look at the Mamba/SSM row: strong fit is "long/streaming sequences," with compact, fixed-size inference memory regardless of transcript length. The trade-off is compressed history — the model may not exactly recall a specific quoted sentence from an hour ago the way attention over a full KV cache could. If exact quoting matters more than cost, look at Transformer instead and accept the growing KV cache.

**Worked example 2: "I need a model that can follow spoken instructions and pick up specific objects with a robot arm."** Look at the VLA row: strong fit is "robotics with language instructions." The trade-off column flags safety and control-frequency limits — check [vision-language-action-models.md](../12-embodied-and-robotics-models/vision-language-action-models.md) for why autoregressive action-token generation is slower than a dedicated low-level controller, and plan for that latency in the control loop.

**Worked example 3: "I want to scale up total model capacity without proportionally increasing the compute cost per token."** Look at the MoE Transformer row: strong fit is "huge capacity at fixed active compute." The trade-off is communication and routing overhead — see [dense-vs-moe.md](dense-vs-moe.md) for the concrete all-to-all communication cost this introduces, which only pays off with enough batch size to keep experts utilized.

There is unlikely to be one universal "post-Transformer" architecture that replaces this whole table. Real systems increasingly combine attention, state-space recurrence, routing, memory, and predictive dynamics inside a single heterogeneous model — see [post-transformer-directions.md](../18-research-frontier/post-transformer-directions.md).

## References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Brohan, A. et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.* [arXiv:2307.15818](https://arxiv.org/abs/2307.15818).
- Behrouz, A., Zhong, P. & Mirrokni, V. (2025). *Titans: Learning to Memorize at Test Time.* [arXiv:2501.00663](https://arxiv.org/abs/2501.00663).

[Back to index](../INDEX.md)
