"""
Component 4: refusal-direction analysis (H4).

Local-only, no GPU/Colab -- reuses Component 2's already-extracted
activations (results/activations/{stage}_pooled.npy + metadata).

For each stage M0-M3, computes a diff-in-means "refusal direction" per
layer:
    direction = mean(activation | quadrant A) - mean(activation | quadrant D)
(A = HarmBench, clearly harmful; D = Alpaca, clearly benign -- the two
least-ambiguous quadrants, kept independent of whichever confound-fix
Component 3 ended up using for its own A-vs-B contrast).

Two things this measures:
  1. Does the direction ITSELF stay stable across training (consistent
     with H4/amplification) or rotate into something new (consistent
     with H1/genuinely richer representation)?
     -> cross-stage cosine similarity of the same layer's direction
        across M0/M1/M2/M3.
  2. Where do quadrant B and C prompts sit relative to that direction,
     and does that shift across stages?
     -> mean projection of each quadrant's activations onto the
        stage's own direction, per layer.
"""
import json
from pathlib import Path

import numpy as np

STAGES = ["M0", "M1", "M2", "M3"]
ACT_DIR = Path("results/activations")
OUT_DIR = Path("results/refusal_direction")
POS_QUADRANT = "A"
NEG_QUADRANT = "D"


def load_stage(stage):
    pooled = np.load(ACT_DIR / f"{stage}_pooled.npy")  # (n_prompts, n_layers, hidden_dim)
    with open(ACT_DIR / f"{stage}_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    quadrants = np.array([row["quadrant"] for row in meta])
    return pooled, quadrants


def diff_in_means_direction(pooled, quadrants, pos_quadrant=POS_QUADRANT, neg_quadrant=NEG_QUADRANT):
    """Returns unit-normalized (n_layers, hidden_dim) direction array."""
    pos_mean = pooled[quadrants == pos_quadrant].mean(axis=0)  # (n_layers, hidden_dim)
    neg_mean = pooled[quadrants == neg_quadrant].mean(axis=0)
    direction = pos_mean - neg_mean
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero on a degenerate layer
    return direction / norms


def cosine_similarity_per_layer(dir_a, dir_b):
    """dir_a, dir_b: (n_layers, hidden_dim), both already unit-normalized."""
    return np.sum(dir_a * dir_b, axis=-1)  # (n_layers,)


def project_onto_direction(pooled, direction):
    """pooled: (n_prompts, n_layers, hidden_dim), direction: (n_layers, hidden_dim).
    Returns (n_prompts, n_layers) scalar projections."""
    return np.einsum("nlh,lh->nl", pooled, direction)


def mean_projection_by_quadrant(pooled, quadrants, direction):
    out = {}
    for q in sorted(set(quadrants.tolist())):
        proj = project_onto_direction(pooled[quadrants == q], direction)
        out[q] = proj.mean(axis=0).tolist()  # per-layer mean, list for JSON
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    directions = {}
    quadrants_by_stage = {}
    pooled_by_stage = {}

    for stage in STAGES:
        pooled, quadrants = load_stage(stage)
        direction = diff_in_means_direction(pooled, quadrants)
        directions[stage] = direction
        quadrants_by_stage[stage] = quadrants
        pooled_by_stage[stage] = pooled
        np.save(OUT_DIR / f"{stage}_direction.npy", direction)
        print(f"{stage}: direction shape {direction.shape}")

    # 1. Cross-stage cosine similarity, per layer, vs M0 (the untrained baseline)
    cosine_vs_m0 = {}
    for stage in STAGES:
        cosine_vs_m0[stage] = cosine_similarity_per_layer(directions["M0"], directions[stage]).tolist()

    # Also the adjacent-stage chain (M0->M1->M2->M3), often more informative than "vs M0"
    cosine_adjacent = {}
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        cosine_adjacent[f"{a}_vs_{b}"] = cosine_similarity_per_layer(directions[a], directions[b]).tolist()

    from src.io_utils import write_json

    write_json(OUT_DIR / "cosine_similarity.json", {"vs_M0": cosine_vs_m0, "adjacent": cosine_adjacent})

    # 2. Mean projection per quadrant per stage, per layer (each stage projected onto ITS OWN direction)
    projections = {}
    for stage in STAGES:
        projections[stage] = mean_projection_by_quadrant(
            pooled_by_stage[stage], quadrants_by_stage[stage], directions[stage]
        )

    write_json(OUT_DIR / "quadrant_projections.json", projections)

    print(f"\nSaved directions, cosine_similarity.json, quadrant_projections.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()