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
    activations_available,
    diff_in_means_direction,
    load_stage,
    project_onto_direction,
)

STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]
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


N_BOOTSTRAP = 1000  # matches bootstrap_direction_stability.py's B=1000
SEED = 0


def bootstrap_bottleneck_layers(pooled, quadrants, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Resample all four quadrants independently, with replacement, and
    recompute BOTH the per-layer direction and the argmax bottleneck layer
    each replicate -- reports which layer wins across resamples of the same
    Cohen's d computation, not just the single argmax on the full sample.
    Answers: is the reported bottleneck layer (e.g. "layer 9") a sharp,
    reliable peak, or would a slightly different sample of prompts have
    picked a different layer nearby? Directly informs how much of the
    9-vs-16 M2-vs-M2_alt gap (README Finding 3) is real vs. argmax noise.

    Returns (a_vs_d_layers, harm_vs_surface_layers): two (n_bootstrap,)
    int arrays of winning layer indices, one per replicate."""
    rng = np.random.default_rng(seed)
    idx_by_quadrant = {q: np.where(quadrants == q)[0] for q in ("A", "B", "C", "D")}

    a_vs_d_layers = np.zeros(n_bootstrap, dtype=int)
    harm_vs_surface_layers = np.zeros(n_bootstrap, dtype=int)
    for i in range(n_bootstrap):
        sample_idx_parts, sample_quadrant_parts = [], []
        for q, idx in idx_by_quadrant.items():
            if len(idx) == 0:
                continue
            sampled = rng.choice(idx, size=len(idx), replace=True)
            sample_idx_parts.append(sampled)
            sample_quadrant_parts.append(np.full(len(sampled), q))
        sample_idx = np.concatenate(sample_idx_parts)
        sample_quadrants = np.concatenate(sample_quadrant_parts)

        d_a_vs_d, d_harm_vs_surface = per_layer_separability(pooled[sample_idx], sample_quadrants)
        a_vs_d_layers[i], _ = find_bottleneck_layer(d_a_vs_d)
        harm_vs_surface_layers[i], _ = find_bottleneck_layer(d_harm_vs_surface)

    return a_vs_d_layers, harm_vs_surface_layers


def summarize_bottleneck_bootstrap(layer_samples, n_layers):
    """layer_samples: (n_bootstrap,) int array of winning-layer indices.
    Reports the full frequency histogram (how concentrated the "winner" is
    across resamples - a sharp single-layer spike vs. a wide, noisy spread
    across many plausible layers), the mode, and a percentile CI on the
    layer index as a compact two-number summary."""
    counts = np.bincount(layer_samples, minlength=n_layers)
    mode = int(np.argmax(counts))
    return {
        "n_bootstrap_replicates": int(len(layer_samples)),
        "mode_layer": mode,
        "mode_frac": float(counts[mode] / len(layer_samples)),
        "ci_low_2.5pct": int(np.percentile(layer_samples, 2.5)),
        "ci_high_97.5pct": int(np.percentile(layer_samples, 97.5)),
        "layer_counts": {int(l): int(c) for l, c in enumerate(counts) if c > 0},
    }


def main():
    out = {}
    print("Bottleneck-layer analysis: Cohen's d per layer, per stage\n")
    for stage in STAGES:
        if not activations_available(stage):
            print(f"=== {stage}: SKIPPED, activations not yet extracted ===")
            continue
        pooled, quadrants = load_stage(stage)
        d_a_vs_d, d_harm_vs_surface = per_layer_separability(pooled, quadrants)
        n_layers = len(d_a_vs_d)

        layer_a_d, effect_a_d = find_bottleneck_layer(d_a_vs_d)
        layer_hs, effect_hs = find_bottleneck_layer(d_harm_vs_surface)

        boot_a_d, boot_hs = bootstrap_bottleneck_layers(pooled, quadrants)
        boot_summary_a_d = summarize_bottleneck_bootstrap(boot_a_d, n_layers)
        boot_summary_hs = summarize_bottleneck_bootstrap(boot_hs, n_layers)

        print(f"=== {stage} ===")
        print(f"  A-vs-D bottleneck layer:            {layer_a_d:>2d}  (Cohen's d = {effect_a_d:+.3f})  "
              f"bootstrap mode {boot_summary_a_d['mode_layer']} "
              f"({boot_summary_a_d['mode_frac']:.0%} of resamples), "
              f"95% CI [{boot_summary_a_d['ci_low_2.5pct']}, {boot_summary_a_d['ci_high_97.5pct']}]")
        print(f"  (A+C)-vs-(B+D) bottleneck layer:     {layer_hs:>2d}  (Cohen's d = {effect_hs:+.3f})  "
              f"bootstrap mode {boot_summary_hs['mode_layer']} "
              f"({boot_summary_hs['mode_frac']:.0%} of resamples), "
              f"95% CI [{boot_summary_hs['ci_low_2.5pct']}, {boot_summary_hs['ci_high_97.5pct']}]")

        out[stage] = {
            "a_vs_d": {
                "per_layer_cohens_d": d_a_vs_d.tolist(),
                "bottleneck_layer": layer_a_d,
                "bottleneck_cohens_d": effect_a_d,
                "bottleneck_bootstrap": boot_summary_a_d,
            },
            "harm_vs_surface_wording": {
                "per_layer_cohens_d": d_harm_vs_surface.tolist(),
                "bottleneck_layer": layer_hs,
                "bottleneck_cohens_d": effect_hs,
                "bottleneck_bootstrap": boot_summary_hs,
            },
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
