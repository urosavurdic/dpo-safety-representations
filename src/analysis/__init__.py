"""
Analysis and measurement pipeline.

Analyzes trained models to extract behavioral and mechanistic signals.

Modules:
- behavioral: Behavioral classification (refusal, compliance, soft-deflection)
- refusal_direction: Extract and summarize refusal direction across layers and stages
- probes: Train and evaluate directional probes on model activations
- causal_ablation: Causal intervention via projection removal and McNemar testing
"""
