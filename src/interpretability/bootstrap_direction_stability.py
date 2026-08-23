"""
Bootstrap direction stability: resample (with replacement) which A/D
prompts contribute to the diff-in-means direction, many times, and check
how consistent the resulting direction is -- answers "is this a real,
stable feature, or an artifact of which particular 50 A/D prompts we
happened to use?"

Reuses eval_refusal_direction.py's already-tested diff_in_means_direction
-- same computation, called on resampled subsets instead of the full set
once.
"""
import json
from pathlib import Path

import numpy as np

from src.analysis.eval_refusal_direction import (
    activations_available, diff_in_means_direction, filter_to_direction_estimation_split,
)

STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]  # M3_direct = M1 + direct DPO, parallel control branch
N_BOOTSTRAP = 1000  # per PROJECT_CONTEXT.md M1D/M3_direct spec: B=1000 replicates
REPORT_LAYERS = None  # None = report all layers (0-28), not a cherry-picked subset
SEED = 0
# README Finding 3 / CLAUDE.md Next-step #3: the "deep layers" where
# direct-DPO branches show a tight bootstrap-stability band (0.98-0.995)
# vs. looser M2-mediated stages (roughly 0.94-0.98). Raw per-replicate sims
# are persisted for these layers only (not all 29) to keep the output file
# a reasonable size while still giving paired_deep_layer_stability_test.py
# everything it needs without rerunning the bootstrap.
DEEP_LAYERS = list(range(16, 29))


def load_stage(stage, act_dir=Path("results/activations")):
    pooled = np.load(act_dir / f"{stage}_pooled.npy")
    with open(act_dir / f"{stage}_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    quadrants = np.array([row["quadrant"] for row in meta])
    splits = np.array([row.get("split") or "" for row in meta])
    return pooled, quadrants, splits


def bootstrap_directions(pooled, quadrants, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Resample quadrant-A and quadrant-D rows independently, with
    replacement, n_bootstrap times; recompute the direction each time.
    Returns (n_bootstrap, n_layers, hidden_dim)."""
    rng = np.random.default_rng(seed)
    a_idx = np.where(quadrants == "A")[0]
    d_idx = np.where(quadrants == "D")[0]

    directions = []
    for _ in range(n_bootstrap):
        a_sample = rng.choice(a_idx, size=len(a_idx), replace=True)
        d_sample = rng.choice(d_idx, size=len(d_idx), replace=True)
        sample_idx = np.concatenate([a_sample, d_sample])
        sample_quadrants = np.array(["A"] * len(a_sample) + ["D"] * len(d_sample))
        direction = diff_in_means_direction(pooled[sample_idx], sample_quadrants)
        directions.append(direction)
    return np.stack(directions)


def stability_sims(bootstrap_dirs, original_direction, layer):
    """Raw (n_bootstrap,) array of cosine similarities between each
    bootstrap replicate's direction and the ORIGINAL full-sample direction,
    at `layer`. Shared core computation for summarize_stability,
    summarize_stability_full, and (for DEEP_LAYERS) the raw arrays persisted
    for paired_deep_layer_stability_test.py -- factored out so all three
    stay consistent instead of recomputing this dot product three ways."""
    orig = original_direction[layer]
    return np.array([np.dot(bd[layer], orig) for bd in bootstrap_dirs])


def summarize_stability(bootstrap_dirs, original_direction, layer):
    """Cosine similarity of each bootstrap direction (at `layer`) to the
    ORIGINAL full-sample direction -- mean and std across replicates.
    Kept as-is (backward compatible, already tested) - summarize_stability_full
    below adds the median/2.5%/97.5% percentiles the original spec also asks for."""
    sims = stability_sims(bootstrap_dirs, original_direction, layer)
    return float(np.mean(sims)), float(np.std(sims))


def summarize_stability_full(bootstrap_dirs, original_direction, layer):
    """Same cosine-similarity-to-original computation as summarize_stability,
    but reports the full distribution: mean, median, std, 2.5th/97.5th
    percentile -- the uncertainty summary PROJECT_CONTEXT.md's bootstrap spec
    asks for, not just mean+std."""
    sims = stability_sims(bootstrap_dirs, original_direction, layer)
    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "std": float(np.std(sims)),
        "ci_low_2.5pct": float(np.percentile(sims, 2.5)),
        "ci_high_97.5pct": float(np.percentile(sims, 97.5)),
    }


def main():
    out = {}
    for stage in STAGES:
        if not activations_available(stage):
            print(f"\n=== {stage}: SKIPPED, activations not yet extracted ===")
            continue
        pooled, quadrants, splits = load_stage(stage)
        pooled, quadrants = filter_to_direction_estimation_split(pooled, quadrants, splits)
        original_direction = diff_in_means_direction(pooled, quadrants)
        bootstrap_dirs = bootstrap_directions(pooled, quadrants)

        n_layers = original_direction.shape[0]
        layers = REPORT_LAYERS if REPORT_LAYERS is not None else range(n_layers)

        print(f"\n=== {stage} ===")
        out[stage] = {}
        for layer in layers:
            stats = summarize_stability_full(bootstrap_dirs, original_direction, layer)
            print(f"  layer {layer}: bootstrap-vs-original cosine sim = "
                  f"{stats['mean']:.4f} (median {stats['median']:.4f}, std {stats['std']:.4f}, "
                  f"95% CI [{stats['ci_low_2.5pct']:.4f}, {stats['ci_high_97.5pct']:.4f}])")
            if layer in DEEP_LAYERS:
                # Task 3 (CLAUDE.md Next steps): persist the raw per-replicate
                # array for deep layers so a formal paired test can be run
                # later without rerunning this (expensive-ish, B=1000 x 9
                # stages) bootstrap.
                stats = {**stats, "raw_sims": stability_sims(bootstrap_dirs, original_direction, layer).tolist()}
            out[stage][layer] = stats

    out_path = Path("results/interpretability/bootstrap_direction_stability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()