# Attention Pattern

Attention is content-addressable interaction. A query vector is compared against a set of key vectors; the comparison scores become weights over a matching set of value vectors.

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

- **Query**: what am I looking for right now?
- **Key**: what information do I represent, for matching purposes?
- **Value**: what content gets retrieved once a match is found?

The `1/sqrt(d_k)` scaling keeps the pre-softmax scores from growing with head dimension `d_k`, which would otherwise push softmax toward a near-one-hot distribution and stall gradients on the non-selected positions. See Attention and Self-Attention for the full derivation and a worked numeric example.

## Where it appears

| Form | Q comes from | K, V come from |
|---|---|---|
| Self-attention | Same sequence | Same sequence |
| Cross-attention | Decoder / query side | Encoder / a different sequence |
| Memory read | Current state | Learned or external memory slots |
| Multimodal fusion | One modality's tokens | Another modality's tokens |

## Cost and trade-off

The advantage is explicit, learned, content-dependent connectivity: any two positions can interact directly, in one step, regardless of distance. The cost is quadratic: with `n` positions, computing and storing all pairwise scores costs `O(n^2)`. At long sequence lengths this cost — not the attention concept itself — is what motivates efficient attention variants, sparse patterns, and state-space alternatives (see Long-Context and Efficient Attention, Recurrence and State Pattern).

## Related patterns

Routing (Routing and Conditional Computation) answers a related but different question — *which parameters run* — while attention answers *which positions interact*. A Mixture-of-Experts router, for instance, is not attention: it produces a hard or soft selection over experts, not a weighted combination of content vectors.

[Back to index](../INDEX.md)
