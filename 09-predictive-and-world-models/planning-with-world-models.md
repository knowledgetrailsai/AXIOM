# Planning with World Models

## One-Minute Explanation

A planner proposes candidate actions or action sequences. A world model predicts the outcome of each. A scoring function picks the sequence whose predicted outcome is closest to the goal. This is the trajectory-optimization pattern behind model-predictive control (MPC).

## Problem It Tries to Solve

A direct policy maps observation to action without comparing alternatives first. It cannot ask "what would happen if I did X instead of Y" before committing. Planning with a world model makes that comparison explicit before any action is executed.

## Core Architectural Idea

The loop has four steps, repeated at every control step:

1. Sample or optimize a set of candidate action sequences.
2. Roll each candidate through the world model to predict a resulting trajectory of future states.
3. Score each predicted trajectory against a goal (e.g. distance in state space, or a learned value function).
4. Execute only the first action of the best-scoring sequence, observe the real outcome, and replan from the new state.

Step 4 — replanning after every single action rather than executing a whole planned sequence blindly — is what makes this model-predictive control rather than open-loop planning. It corrects for world-model error using fresh real observations at every step.

**Worked example.** Suppose the planner samples 3 candidate 2-step action sequences and rolls each through the world model, producing a predicted reward for reaching the goal:

| Candidate sequence | Predicted reward |
|---|---|
| A: [turn-left, forward] | 0.62 |
| B: [forward, forward] | 0.81 |
| C: [turn-right, forward] | 0.45 |

`argmax(0.62, 0.81, 0.45) = B`. The planner executes only the first action of B ("forward"), observes the real next state, and replans from there for the next step.

## Information Flow

```mermaid
flowchart LR
    S[Current state] --> G[Sample candidate action sequences]
    G --> WM[World model rollout]
    WM --> F[Predicted future states]
    F --> SC[Score against goal]
    SC --> B[Select best sequence]
    B --> E[Execute first action]
    E --> S
```

## Components

| Component | Role |
|---|---|
| Candidate generator | Samples or optimizes a set of action sequences (random shooting, cross-entropy method, gradient-based search) |
| World model | Predicts the resulting state trajectory for each candidate, in latent or observation space |
| Scoring function | Assigns a reward/distance score to each predicted trajectory relative to the goal |
| Replanning loop | Executes only the first action, then repeats planning from the new real state |

## Computational Characteristics

| Dimension | Notes |
|---|---|
| training parallelism | Only the world model itself needs training; planning is an inference-time search, not a training procedure |
| sequence scaling | Cost scales with (number of candidates) × (planning horizon) world-model calls per control step |
| total parameters | Determined entirely by the underlying world model (see [what-is-a-world-model.md](what-is-a-world-model.md)) |
| active parameters | Same world-model parameters reused for every candidate rollout |
| persistent inference state | Current real state; candidate trajectories are discarded after each replanning step |
| communication | Single-node in most implementations; candidate rollouts can be batched or run in parallel across devices |

## Strengths

- Explicit counterfactual evaluation before acting, rather than a single unexamined action choice.
- One learned dynamics model can be reused across many different goals — only the scoring function changes.
- Replanning at every step corrects for model error using real feedback, which is the core idea behind MPC.

## Limitations and Failure Modes

- Planning multiplies inference cost by (candidates × horizon) world-model calls per control step.
- A planner searching hard enough can find and exploit systematic errors in the world model, producing action sequences that look good to the model but fail in reality.
- Search cost grows combinatorially with horizon length and action-space size unless candidates are sampled or pruned intelligently.

## Architecture vs Training Objective

The world model's architecture (see [what-is-a-world-model.md](what-is-a-world-model.md)) is trained once, independent of any particular planning problem. The planning loop itself — how candidates are sampled, how many are evaluated, how scoring works — is an inference-time algorithm layered on top, not part of the trained architecture.

## When to Use It

Use planning with a world model when the action space is well-defined, a reasonably accurate dynamics model is available, and the cost of extra inference-time search is acceptable relative to the value of avoiding a bad action.

## When Not to Use It

Do not use it when control latency budgets are too tight for multiple rollout evaluations per step, or when the world model's prediction error is large enough that the planner would mostly be exploiting model mistakes rather than finding genuinely good actions.

## Comparison with Alternatives

Generator-verifier reasoning (see [generator-verifier.md](../11-reasoning-oriented-architectures/generator-verifier.md)) follows a structurally identical propose → evaluate → select loop, applied to discrete solutions instead of continuous action trajectories.

## Representative Models

Model-predictive control paired with V-JEPA 2 (see [v-jepa-2.md](v-jepa-2.md)); Dreamer-style latent imagination planners.

## References

- Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).

[Back to index](../INDEX.md)
