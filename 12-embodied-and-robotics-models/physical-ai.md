# Physical AI

Physical AI is not one architecture. It is a stack, and different systems draw the boundaries between its stages differently:

```text
perception encoder
→ multimodal representation (vision, language, proprioception fused)
→ world state / memory
→ world model or predictive dynamics
→ planner / value model
→ policy / action decoder
→ controller (low-level, high-frequency)
```

The central open design question is how much of this stack should be one end-to-end trained model versus separate, independently-built modules. A VLA model (see [vision-language-action-models.md](vision-language-action-models.md)) collapses perception, representation, and policy into a single trained model that outputs action tokens directly. A world-model-based stack (see [world-models-for-robotics.md](world-models-for-robotics.md)) keeps the dynamics-prediction and planning stages explicit and separate from the perception encoder.

## Constraints that are first-class here but secondary elsewhere in this repository

| Constraint | Why it matters specifically for physical AI |
|---|---|
| Control-loop latency | A prediction that is correct but arrives too late is useless for real-time control |
| Partial observability | Sensors rarely capture full physical state (occlusion, sensor noise, unmodeled contact forces) |
| Changing dynamics | Wear, payload changes, and environmental variation shift the true dynamics away from what was modeled at training time |
| Irreversible actions | Unlike regenerating a wrong text output, a dropped or broken object cannot be undone |
| Edge compute limits | On-robot inference is often constrained to embedded or edge hardware, far below datacenter GPU budgets |
| Calibration | Camera-to-robot and sensor-to-actuator coordinate transforms must be accurate for any learned model's outputs to map correctly onto physical motion |
| Safety | An out-of-distribution action can cause physical harm, not just a wrong answer |

None of these constraints are architectural in the sense of "which mechanism computes an output" — but they directly determine which architectural choices (a large VLA backbone vs. a small dedicated policy, latent planning vs. generative rollout) are even feasible for a given robot and task.

## References

- Brohan, A. et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.* [arXiv:2307.15818](https://arxiv.org/abs/2307.15818).

[Back to index](../INDEX.md)
