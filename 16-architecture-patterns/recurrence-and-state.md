# Recurrence and State Pattern

The abstract recurrence is:

`h_t = f(h_{t-1}, x_t)`

A fixed-size state `h_t` is carried forward and updated by combining it with the current input. Everything the model "remembers" about the past has to fit inside `h_t` — this is a hard compression constraint, not a soft one.

## Concrete instances

- **Vanilla RNN**: `h_t = tanh(W_h h_{t-1} + W_x x_t + b)`. Repeated multiplication by the same Jacobian across timesteps causes vanishing/exploding gradients (see RNN, LSTM and GRU).
- **LSTM**: adds a gated cell state, `c_t = f_t * c_{t-1} + i_t * c~_t`, where the forget gate `f_t` lets old state pass through nearly unchanged when `f_t ≈ 1` — an additive gradient shortcut, structurally similar to a residual connection's `+1` term.
- **SSM / Mamba**: state update `h_t = A h_{t-1} + B x_t`, with `A` and `B` either fixed or (in Mamba's *selective* SSM) themselves a function of the current input `x_t`, letting the model dynamically decide what to retain per step rather than using fixed dynamics (see Mamba and SSM Families).
- **RWKV, xLSTM**: modern architectures that keep the recurrence form `h_t = f(h_{t-1}, x_t)` but redesign `f` for better long-range retention and hardware-parallel training.

## Strength and weakness

The strength is compact, constant-size (`O(1)` in sequence length) streaming state — a fixed memory footprint regardless of how long the sequence has run. The weakness is information compression: anything not retained in `h_t` is not recoverable later, unlike attention's explicit, addressable, growing KV cache that in principle keeps every past position directly accessible (at `O(n)` memory cost instead of `O(1)`).

## Where this shows up

RNNs, LSTMs, GRUs, SSMs, Mamba, RWKV, xLSTM, recurrent test-time reasoning loops, and world-dynamics models that roll a latent state forward through simulated time all instantiate this same recurrence form, differing mainly in how expressive and trainable `f` is made.

[Back to index](../INDEX.md)
