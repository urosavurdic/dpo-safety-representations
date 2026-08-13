"""
Mechanistic interpretability and explanation.

Post-hoc analysis to understand *how* and *why* the refusal direction and causal
intervention work. Based on mechanistic interpretability literature.

Modules:
- per_layer_analysis: Attribution analysis—which layers contribute most to the effect
- alpha_scaling: Interpolation analysis—effect magnitude as a function of projection scale
- direction_stability: Direction drift and alignment across model checkpoints
- utils: Shared utilities for interpretability analysis
"""
