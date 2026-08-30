# Terminology

| Term | Meaning |
|---|---|
| action chunking | predicting a fixed-length block of future actions in one forward pass, instead of one action per step |
| action tokenization | discretizing a continuous action value into a bin/token index, so a discrete-token model can generate it |
| active parameters | parameters actually executed for a token/example |
| adaptive computation | letting a model spend a variable amount of compute per input, e.g. via early exit |
| associative memory loss | a loss measuring how well a memory module's current state predicts/retrieves an input, used to drive test-time memory updates |
| attention | query-key weighted aggregation of values |
| autoregressive | predicts each output unit from prior units |
| backbone | main reusable neural architecture |
| byte-level model | processes raw bytes instead of tokens from a fixed vocabulary |
| causal mask | blocks access to future sequence positions |
| chain-of-thought | a reasoning trace expressed as generated intermediate tokens before a final answer |
| conditional computation | input-dependent selection of executed modules |
| context encoder | in JEPA, the encoder that maps the visible/context portion of an input to an embedding |
| contrastive loss | a loss that pulls representations of related inputs together and pushes unrelated ones apart |
| convolution | shared local filter across positions |
| cross-attention | queries from one stream attend to keys/values from another |
| decay (memory) | a term in a memory update rule that fades old content over time |
| dense model | most relevant parameters execute for every input |
| diffusion | generative process that learns to reverse corruption/noise |
| discretization | mapping a continuous value to one of a fixed set of bins/tokens |
| dual encoder | two separate encoders (e.g. for two modalities) trained so their outputs land in a shared, comparable embedding space |
| dynamic depth | varying how many layers execute per input |
| early exit | halting computation at an intermediate layer once a confidence threshold is met |
| ELBO | evidence lower bound; the tractable objective optimized in place of the true (intractable) data likelihood in latent-variable generative models |
| EMA (exponential moving average) | an update rule that slowly tracks another set of weights, `θ ← τ·θ + (1-τ)·θ_source` |
| embedding | continuous learned vector representation |
| encoder | maps input into representation |
| expert | specialist subnetwork in MoE |
| FFN | feed-forward network, often an MLP inside Transformer blocks |
| flow matching | learns a vector field transporting distributions |
| generator-verifier | an architecture pattern where a generator proposes candidates and a separate verifier scores them |
| GQA | grouped-query attention |
| halting mechanism | a learned or fixed rule deciding when to stop an iterative or adaptive-depth computation |
| JEPA | joint embedding predictive architecture |
| KV cache | stored key/value activations for autoregressive attention |
| latent | internal learned representation |
| latent action model | a model that infers a compact action-like variable from observed transitions, without ground-truth action labels |
| latent reasoning | performing intermediate reasoning computation in hidden-state space rather than as generated tokens |
| load balancing loss | an auxiliary MoE training loss that encourages tokens to be routed roughly evenly across experts |
| majority vote / self-consistency | selecting the most common answer across multiple sampled reasoning paths |
| model-predictive control (MPC) | replanning at every step by choosing the best predicted action sequence and executing only its first action |
| MoE | mixture of experts |
| MQA | multi-query attention |
| neural memory module | a small trainable network used as a memory store, potentially updated at test time |
| patch (BLT) | a variable-length group of bytes formed dynamically based on local predictability |
| perception-action loop | the closed loop of observing, deciding, acting, and re-observing that defines an embodied agent |
| predictor (JEPA) | the component that maps a context embedding to a predicted target embedding |
| quantization | reducing numeric precision of weights/activations (or, for actions, mapping continuous values to discrete bins) |
| recurrent state | hidden state carried across steps |
| residual connection | shortcut adding input to transformed output |
| RoPE | rotary position embedding |
| router | chooses which expert/module executes |
| self-attention | attention within one sequence |
| self-supervised learning | training on labels derived automatically from the data itself, with no human annotation |
| sparse model | activates a subset of connections/parameters |
| speculative decoding | using a smaller draft model to propose multiple tokens, verified in one pass by the larger model, to speed up autoregressive generation |
| SSM | state-space model |
| stop-gradient | preventing gradients from flowing through a branch of computation, used to avoid representation collapse |
| surprise signal | a prediction-error measure used to decide how much to update a test-time memory module |
| target encoder | in JEPA, the (usually EMA-updated) encoder that produces the embedding a predictor is trained to match |
| test-time compute | extra inference computation after training |
| test-time learning | memory/parameter adaptation during inference |
| trajectory optimization | choosing an action sequence by rolling candidates through a model and scoring predicted outcomes |
| VLA | vision-language-action model |
| world model | predictive model of environment dynamics |

[Back to repository index](../INDEX.md)
