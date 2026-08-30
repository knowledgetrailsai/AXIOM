# MoE Systems Trade-offs

## One-Minute Explanation

An MoE layer's FLOPs-per-token count looks cheap on paper, but real latency depends on how tokens physically move between devices. When experts live on different accelerators, every MoE layer requires an all-to-all exchange: dispatch tokens to their assigned expert's device, then gather results back. That exchange, not the matrix multiply itself, is often what determines wall-clock throughput.

## Problem It Tries to Solve

A FLOPs count assumes compute is the bottleneck. Once experts are spread across a cluster, network bandwidth and dispatch latency compete directly with the compute savings MoE was supposed to deliver. A system that ignores this can end up slower than a dense model with equivalent active FLOPs.

## Core Architectural Idea

Expert parallelism places different experts on different accelerators (or groups of accelerators). For each MoE layer, the runtime:

1. Computes routing decisions locally (cheap — router is small).
2. Dispatches each token's hidden state to the device(s) hosting its top-k experts (an all-to-all communication step).
3. Runs the expert FFN on the receiving device.
4. Gathers results back to the token's originating device and combines them (a second all-to-all).

Two all-to-all operations per MoE layer means communication cost scales with (batch size × hidden dimension × k), not with the number of experts N directly. This is why increasing N (to add capacity) is nearly free in communication terms, while increasing batch size or hidden dimension directly increases the data volume that must cross the network every layer.

**Batch size sensitivity.** If a batch is small, each expert receives few tokens, so the useful compute per dispatch is small relative to the fixed latency of initiating and completing an all-to-all. Larger batches amortize that fixed cost — this is why MoE inference commonly needs batching (many concurrent requests) to reach the FLOPs advantage the architecture promises on paper.

## Information Flow

```mermaid
flowchart LR
    L[Local: router on each device] --> D1[Dispatch: all-to-all send tokens to expert device]
    D1 --> E[Expert FFN compute, per device]
    E --> D2[Combine: all-to-all gather results back]
    D2 --> O[Output on originating device]
```

## Components

| Component | Role |
|---|---|
| Expert-parallel placement | Assignment of experts to devices; determines which dispatch is local vs. cross-device |
| All-to-all dispatch | Network operation sending each token to its chosen expert's device |
| All-to-all combine | Network operation gathering expert outputs back to the token's origin |
| Capacity factor | Bounds per-expert token count per batch, which bounds the size of each dispatch |
| Batch scheduler | Groups enough concurrent tokens to amortize dispatch latency against useful compute |

## Computational Characteristics

| Dimension | Notes |
|---|---|
| training parallelism | Combines data parallelism (across batches) with expert parallelism (across experts); the two must be coordinated in the parallelism plan |
| sequence scaling | Independent of MoE communication cost; that cost scales with token count in the batch, not sequence length per se |
| total parameters | Same as base MoE — grows with N |
| active parameters | Same as base MoE — set by k |
| persistent inference state | Unaffected; determined by the attention/recurrence mechanism |
| communication | Two all-to-all operations per MoE layer per forward pass; volume scales with batch size × hidden dim × k, not with N |

## Strengths

High capacity per active FLOP, as with base MoE. This pattern is specifically well suited to very large clusters where enough devices exist to host many experts and enough interconnect bandwidth exists to make dispatch cheap relative to compute.

## Limitations and Failure Modes

At small batch sizes, dispatch latency can exceed the compute time it's supposed to enable, making the sparse layer slower in wall-clock terms than an equivalent dense layer would have been. All-to-all communication volume and latency scale with the interconnect topology; a system with weaker inter-device bandwidth (e.g. across nodes rather than within one) pays disproportionately for the same MoE architecture. Uneven routing (see [load-balancing-and-specialization.md](load-balancing-and-specialization.md)) means some devices finish their expert compute early and idle while waiting on others — a straggler problem layered on top of the communication cost.

## Architecture vs Training Objective

Expert parallelism is a systems/deployment decision about *where* the architecturally-defined experts live, not a change to the model's forward-pass definition. The same trained MoE weights can be served with different expert-to-device placements, with very different latency, and identical outputs.

## When to Use It

Very large clusters with high-bandwidth interconnect (e.g. within a single high-speed fabric) and consistently large batch sizes, where the FLOPs savings of sparsity translate into real throughput gains once amortized over enough tokens per dispatch.

## When Not to Use It

Single-machine or low-bandwidth-interconnect deployments, or workloads with persistently small batch sizes (e.g. single-user, low-concurrency serving) — in both cases, dispatch overhead is likely to erase the FLOPs advantage.

## Comparison with Alternatives

Dense models are operationally simpler: no routing, no all-to-all, no capacity tuning, no straggler risk. MoE only wins when the system — cluster topology, batch scheduling, capacity tuning — actually converts the FLOPs savings into throughput; otherwise a dense model at the active-parameter scale can be the faster real-world choice despite doing more total arithmetic per token in isolation.

## Representative Models

Switch Transformer and Mixtral 8x7B are both deployed with expert-parallel serving in practice; their published efficiency claims assume batched serving with enough concurrent tokens to amortize dispatch cost.

## References

- Fedus, W. et al. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).
- Jiang, A.Q. et al. (2024). *Mixtral of Experts.* [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* [arXiv:1701.06538](https://arxiv.org/abs/1701.06538).

[Back to index](../INDEX.md)
