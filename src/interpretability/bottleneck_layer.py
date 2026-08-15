"""
Bottleneck-layer analysis: at which layer is the diff-in-means refusal
direction's separation of the quadrants strongest?

Motivation: the retired naive-CV-accuracy probe metric (see
summarize_probe_findings.py's docstring) saturates near 1.0 at almost every
layer for every stage, including untrained M0 - not usable for finding a
"most separable" layer. This reuses the SAME per-layer direction/projection
machinery Component 4 already validated (eval_refusal_direction.py:
load_stage, diff_in_means_direction, project_onto_direction) instead of a
second probing implementation, and reports a continuous effect size
(Cohen's d) per layer, which doesn't have the saturation problem a
thresholded classifier does.

Two separability questions, both meaningful:
  1. A vs D (the two quadrants the direction is DEFINED from): how cleanly
     does this layer's own direction separate the data it was built on?
  2. (A+C) vs (B+D) - true harmfulness vs surface wording: does this layer's
     direction distinguish REAL threat (A, C) from prompts that only LOOK
     harmful or LOOK benign (B mimics harmful-sounding-but-safe, C mimics
     benign-sounding-but-harmful)? This is the sharper, more interesting
     question - a layer could separate A-vs-D trivially while barely
     separating the disguised/ambiguous cases at all.

The "bottleneck layer" is the argmax of each effect size, per stage.
"""
import json
from pathlib import Path

import numpy as np

from src.analysis.eval_refusal_direction import (
    diff_in_means_direction,
    load_stage,
    project_onto_direction,
)

STAGES = ["M0", "M1", "M2", "M3", "M3_direct"]
OUT_PATH = Path("results/interpretability/bottleneck_layer.json")


def cohens_d(x, y):
    """Standardized mean difference. Pooled std; returns 0.0 (not NaN) for a
    degenerate all-identical-value layer instead of dividing by zero."""
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled_std)


def per_layer_separability(pooled, quadrants):
    """Returns two (n_layers,) arrays: Cohen's d for A-vs-D and for
    (A+C)-vs-(B+D), at each layer, using that stage's OWN per-layer
    diff-in-means direction (not a single global direction)."""
    direction = diff_in_means_direction(pooled, quadrants)  # (n_layers, hidden_dim)
    proj = project_onto_direction(pooled, direction)  # (n_prompts, n_layers)

    n_layers = proj.shape[1]
    d_a_vs_d = np.zeros(n_layers)
    d_harm_vs_surface = np.zeros(n_layers)

    is_a, is_d = quadrants == "A", quadrants == "D"
    is_harm, is_surface = np.isin(quadrants, ["A", "C"]), np.isin(quadrants, ["B", "D"])

    for layer in range(n_layers):
        d_a_vs_d[layer] = cohens_d(proj[is_a, layer], proj[is_d, layer])
        d_harm_vs_surface[layer] = cohens_d(proj[is_harm, layer], proj[is_surface, layer])

    return d_a_vs_d, d_harm_vs_surface


def find_bottleneck_layer(effect_sizes):
    """argmax of |effect size| -- separability could in principle be
    strongest in either direction sign, magnitude is what matters."""
    idx = int(np.argmax(np.abs(effect_sizes)))
    return idx, float(effect_sizes[idx])


def main():
    out = {}
    print("Bottleneck-layer analysis: Cohen's d per layer, per stage\n")
    for stage in STAGES:
        pooled, quadrants = load_stage(stage)
        d_a_vs_d, d_harm_vs_surface = per_layer_separability(pooled, quadrants)

        layer_a_d, effect_a_d = find_bottleneck_layer(d_a_vs_d)
        layer_hs, effect_hs = find_bottleneck_layer(d_harm_vs_surface)

        print(f"=== {stage} ===")
        print(f"  A-vs-D bottleneck layer:            {layer_a_d:>2d}  (Cohen's d = {effect_a_d:+.3f})")
        print(f"  (A+C)-vs-(B+D) bottleneck layer:     {layer_hs:>2d}  (Cohen's d = {effect_hs:+.3f})")

        out[stage] = {
            "a_vs_d": {
                "per_layer_cohens_d": d_a_vs_d.tolist(),
                "bottleneck_layer": layer_a_d,
                "bottleneck_cohens_d": effect_a_d,
            },
            "harm_vs_surface_wording": {
                "per_layer_cohens_d": d_harm_vs_surface.tolist(),
                "bottleneck_layer": layer_hs,
                "bottleneck_cohens_d": effect_hs,
            },
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
