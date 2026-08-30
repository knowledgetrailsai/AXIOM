# Architecture Selection Method

1. **Characterize input structure** — spatial, sequential, graph, multimodal, action-conditioned.
2. **Characterize output** — embedding, sequence, media, future state, action.
3. **Set compute constraints** — training budget, latency, memory, bandwidth, concurrency.
4. **Separate training from inference** — some architectures train in parallel but serve with large state; others do the reverse.
5. **Pick a baseline** — every architecture claim needs a comparison regime.
6. **Track total and active parameters separately** — especially for MoE.
7. **Track state separately from weights** — KV cache, recurrent state and learned memory behave differently.

A practical architecture decision matrix should include FLOPs, active parameters, persistent state, communication, sequential depth and hardware utilization.
