# Adaptive and Test-Time Learning

Test-time learning means a model changes something about itself — its state, or even its own weights — while it is actively processing a query, rather than only during a separate training phase. Several architectural surfaces support this:

- **Neural-memory updates.** A module's weights update from a surprise signal during inference, as in Titans (see [titans-test-time-memory.md](../10-memory-and-adaptive-computation/titans-test-time-memory.md)).
- **Recurrent hidden state.** A fixed-size state is updated at every step, carrying information forward within a session (see [external-and-recurrent-memory.md](../10-memory-and-adaptive-computation/external-and-recurrent-memory.md)).
- **Dynamic depth.** How much computation runs varies per input, decided at inference time (see [adaptive-computation-and-dynamic-depth.md](../10-memory-and-adaptive-computation/adaptive-computation-and-dynamic-depth.md)).
- **Test-time optimization.** A small optimization loop runs at inference to adapt part of the model to the current input or task before producing an output.
- **Routing.** Which parameters execute is decided per input at inference time, as in MoE (see [dense-vs-moe.md](../15-architecture-comparisons/dense-vs-moe.md)).
- **Fast weights.** A secondary, quickly-adapting weight matrix is computed from recent context and combined with the slow, trained weights.

## Open Engineering Questions

Whether a model that updates its own weights mid-session stays stable over long sessions, without either forgetting earlier useful updates or drifting into degraded behavior, is not established by public research as of this writing for most of these mechanisms. Concretely unresolved, as of this writing:

- **Stability.** Does the surprise-driven update rule in a system like Titans stay bounded over arbitrarily long input streams, or can it be driven to a degenerate state by adversarial or unusual input sequences?
- **Forgetting.** When old memory content decays to make room for new content, is there a principled way to guarantee that nothing task-critical is lost, or is this purely an empirical tuning problem per deployment?
- **Privacy.** A model whose weights change based on a specific user's session raises questions about whether information from that session could leak into later sessions if the memory or weight updates are not properly isolated per user.
- **Reproducibility.** The same prompt processed twice can produce different outputs if test-time state differs between the two runs — this breaks a common assumption in evaluation and debugging that a fixed prompt has a deterministic-enough behavior to reason about.
- **Reset semantics.** What exactly gets reset between sessions or requests, and what persists, has to be an explicit design decision — leaving it implicit risks state leaking across users or requests.
- **Concurrent serving.** A serving system built for stateless requests has to be re-architected to safely hold and isolate mutable per-session state for many concurrent users at once.

## References

- Behrouz, A., Zhong, P. & Mirrokni, V. (2025). *Titans: Learning to Memorize at Test Time.* [arXiv:2501.00663](https://arxiv.org/abs/2501.00663).

[Back to index](../INDEX.md)
