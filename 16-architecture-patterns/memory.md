# Memory Pattern

| Type | Lives in | Size behavior | Example |
|---|---|---|---|
| Context activations | Current forward context | Grows with input length, discarded after the call | Transformer's live hidden states |
| Cache | Saved activations | Grows linearly with sequence length: `2 * layers * kv_heads * head_dim * seq_len * batch * bytes` | KV cache (see MQA, GQA and KV Cache) |
| Recurrent state | Compact fixed-size state | Constant size `O(1)` regardless of history length | RNN / SSM hidden state, `h_t = f(h_{t-1}, x_t)` |
| Memory tokens | Learned token slots | Fixed count, set at design time | Memory-augmented Transformers with extra learnable slots |
| Neural memory | Adaptable module state | Updated at test time via its own learned update rule | Titans-like test-time-learning memory modules |
| External store | Outside the model entirely | Unbounded, limited only by storage infrastructure | RAG / vector database |

## The size vs. addressability trade-off

Moving down this table roughly trades size for addressability. A KV cache keeps every past token individually addressable but grows without bound. A recurrent state is bounded and cheap but compresses history irreversibly — anything not retained in `h_t` is gone. An external store (RAG) has no size limit but requires a separate retrieval step and is not part of the model's own differentiable memory at all.

## System vs. model architecture

The external-store row is primarily **system architecture**, not backbone architecture — a vector database and retrieval pipeline sit outside the model's own forward computation graph, unlike the other five rows, which are properties of the model itself (see the Model Architecture vs System Architecture distinction in `00-navigation-and-methodology/knowledge-map.md`).

[Back to index](../INDEX.md)
