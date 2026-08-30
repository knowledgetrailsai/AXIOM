# Hierarchy and Modularity

## Hierarchy: operating at multiple scales

Hierarchy lets a model process the same input at more than one resolution or timescale simultaneously, rather than treating every position or timestep uniformly:

- **Spatial hierarchy**: image pyramids, or a hierarchical Vision Transformer (e.g. Swin) that merges patches into progressively coarser tokens across stages, recovering CNN-like multiscale structure inside an attention-based model (see Vision Transformers).
- **Sequence hierarchy**: byte-level patching (Byte Latent Transformer) groups raw bytes into dynamically-sized patches before applying a Transformer, changing the effective granularity the sequence model operates over.
- **Attention hierarchy**: combining local (fine-grained, short-range) attention with global (coarse-grained, long-range) attention in the same model, as in local+global sparse attention patterns (see Long-Context and Efficient Attention).
- **Memory hierarchy**: short-term context (current activations) versus long-term memory (recurrent state, external store) — see the Memory Pattern table.

## Modularity: separating by function

Modularity splits computation across specialized subnetworks rather than one homogeneous stack:

- **MoE experts**: several FFN subnetworks, a router selecting which run per token (see Routing and Conditional Computation).
- **Modality encoders**: separate encoder networks per input modality (text, image, audio) feeding into a shared backbone.
- **Generator/verifier pairs**: one module proposes candidates, a separate module scores or filters them — common in reasoning and search-augmented systems.
- **Planner/world-model modules**: a world model predicts consequences of candidate actions; a separate planner or policy selects among them using those predictions.

## Why this is becoming more important

A single homogeneous stack (one uniform Transformer block repeated `L` times) is simple but treats every part of every input identically. As systems take on multimodal input, long-range dependencies at multiple timescales, and multi-step reasoning, both patterns let different parts of the computation specialize — by scale (hierarchy) or by function (modularity) — instead of forcing one uniform mechanism to do everything equally well.

[Back to index](../INDEX.md)
