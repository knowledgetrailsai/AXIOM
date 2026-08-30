# Edge AI Architecture

Edge deployment inverts the usual assumption of near-unlimited data-center memory and power. Every architectural choice has to answer to a small, fixed budget instead.

## The constraints

- **RAM/VRAM.** A typical phone has on the order of 6–12 GB total system RAM, of which only a fraction — often a few GB — is realistically available to a single on-device model alongside the OS and other apps. A laptop or edge accelerator budget is larger but still fixed and shared.
- **Power and thermal envelope.** Mobile and embedded chips throttle under sustained load; sustained peak compute is rarely available for long, unlike a data-center accelerator with active cooling.
- **Latency.** Interactive on-device use cases (voice assistants, camera-based features) usually need results in well under a second, with no option to queue or batch across users the way a server can.
- **Intermittent connectivity.** On-device inference needs to work with no network at all, ruling out any architecture that depends on a server-side component.
- **Privacy.** Keeping data on-device is often a hard requirement, not just an optimization, which rules out designs that need to phone home for part of the computation.
- **Accelerator support.** Mobile/embedded NPUs support a narrower set of operations efficiently than a data-center GPU; an architecture using an unsupported op falls back to a much slower code path.

## Worked memory example

A 7B-parameter model at fp16 needs about 14 GB just for weights (see [quantization-and-architecture.md](quantization-and-architecture.md) for the full memory table), which alone exceeds a typical phone's entire usable RAM budget for a single app. Quantized to INT4, the same model needs about 3.5 GB — within reach of a device with several GB free, though still a large fraction of what's typically available. This is why edge deployment essentially never uses full-precision weights: the memory math simply does not work without quantization, distillation, or both applied together.

## Common design responses

| Response | What it does |
|---|---|
| Compact CNNs | Convolutional locality is cheap and well-supported by mobile NPUs; a good fit when the task is inherently spatial (vision) |
| Small GQA Transformers | Grouped-query attention (see [03-attention-and-transformers/mqa-gqa-and-kv-cache.md](../03-attention-and-transformers/mqa-gqa-and-kv-cache.md)) shrinks the KV cache, directly addressing a memory constraint that hits small devices hardest |
| Recurrent/SSM backbones | Constant-size inference state (see [06-state-space-and-recurrent-alternatives/state-space-models-and-s4.md](../06-state-space-and-recurrent-alternatives/state-space-models-and-s4.md)) avoids a growing cache altogether, valuable when RAM headroom is thin |
| Quantization | INT8/INT4 weights (see [quantization-and-architecture.md](quantization-and-architecture.md)) — usually mandatory rather than optional at edge scale, per the worked example above |
| Distillation | Trains a genuinely smaller model to imitate a larger one, reducing both memory and active compute rather than just memory |
| Modality-specific front ends | Handle the sensor input format directly (e.g. a compact audio front end) rather than relying on a generic, larger pipeline |

## The hardware-support caveat

Irregular sparsity (e.g. MoE routing, see [05-sparse-and-mixture-of-experts/mixture-of-experts.md](../05-sparse-and-mixture-of-experts/mixture-of-experts.md)) is only useful if the target hardware can actually skip the unused computation. A mobile NPU built around fixed-shape dense matrix multiplies may not benefit at all from a theoretically sparse model, and the routing/dispatch overhead can make it slower in practice than an equivalently-sized dense model — the FLOPs savings only matter if the hardware and software stack can realize them.

## References

- Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).
- Ainslie, J. et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).

[Back to index](../INDEX.md)
