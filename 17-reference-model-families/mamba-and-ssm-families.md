# Mamba and SSM Families

A useful lineage: **structured SSM (S4) → selective state updates (Mamba) → state-space duality (Mamba-2) → ongoing selective-SSM research**.

## Architectural lesson

All members of this family use the recurrence form `h_t = A h_{t-1} + B x_t` (see Recurrence and State Pattern), giving `O(n)` training cost and `O(1)` inference state, in contrast to attention's `O(n^2)` training cost and `O(n)` growing KV cache.

- **S4** (Gu et al., 2021) uses a fixed, structured `A` matrix (HiPPO initialization) chosen to preserve long-range history well, with `A` and `B` constant across the sequence — the state update dynamics do not depend on the input.
- **Mamba** (Gu & Dao, 2023) makes `A` and `B` functions of the current input `x_t` — a *selective* SSM — so the model can dynamically decide, per token, what to retain or discard, closing much of the quality gap with attention while keeping linear-time training via a hardware-aware parallel scan.
- **Mamba-2** (Dao & Gu, 2024) establishes a formal duality between selective SSMs and a restricted form of linear attention, unifying the two mechanisms theoretically and enabling further hardware efficiency gains.

## Representative models

| Model | Recurrence type | State dynamics | Notable property |
|---|---|---|---|
| S4 (Gu et al., 2021) | Structured, fixed SSM | Input-independent `A`, `B` | HiPPO-initialized long-range memory |
| Mamba (Gu & Dao, 2023) | Selective SSM | Input-dependent `A`, `B`, `Δ` | Hardware-aware parallel scan, linear-time training |
| Mamba-2 (Dao & Gu, 2024) | Selective SSM, SSD form | Input-dependent, structured for matmul efficiency | State-space/attention duality |

## References

- Gu, A., Goel, K. & Ré, C. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* [arXiv:2111.00396](https://arxiv.org/abs/2111.00396).
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Dao, T. & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2).* [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).

[Back to index](../INDEX.md)
