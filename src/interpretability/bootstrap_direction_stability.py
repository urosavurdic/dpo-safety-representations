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

from src.analysis.eval_refusal_direction import diff_in_means_direction

STAGES = ["M0", "M1", "M2", "M3"]
N_BOOTSTRAP = 200
REPORT_LAYERS = [7, 14, 21, 28]
SEED = 0


def load_stage(stage, act_dir=Path("results/activations")):
    pooled = np.load(act_dir / f"{stage}_pooled.npy")
    with open(act_dir / f"{stage}_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    quadrants = np.array([row["quadrant"] for row in meta])
    return pooled, quadrants


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


def summarize_stability(bootstrap_dirs, original_direction, layer):
    """Cosine similarity of each bootstrap direction (at `layer`) to the
    ORIGINAL full-sample direction -- mean and std across replicates."""
    orig = original_direction[layer]
    sims = [np.dot(bd[layer], orig) for bd in bootstrap_dirs]
    return float(np.mean(sims)), float(np.std(sims))


def main():
    out = {}
    for stage in STAGES:
        pooled, quadrants = load_stage(stage)
        original_direction = diff_in_means_direction(pooled, quadrants)
        bootstrap_dirs = bootstrap_directions(pooled, quadrants)

        print(f"\n=== {stage} ===")
        out[stage] = {}
        for layer in REPORT_LAYERS:
            mean_sim, std_sim = summarize_stability(bootstrap_dirs, original_direction, layer)
            print(f"  layer {layer}: bootstrap-vs-original cosine sim = {mean_sim:.4f} ± {std_sim:.4f}")
            out[stage][layer] = {"mean_cosine_sim": mean_sim, "std_cosine_sim": std_sim}

    out_path = Path("results/interpretability/bootstrap_direction_stability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()