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

STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]
# The TRUE sequential trajectory is only M0->M1->M2->M3 - M3_direct branches
# from M1 directly (parallel control branch, not a "next stage" after M3).
# Kept separate from STAGES so the adjacent-chain table never mislabels a
# M3_direct comparison as if it were part of the sequential narrative -
# see PROJECT_CONTEXT.md/HANDOFF.md M3_direct notes.
SEQUENTIAL_STAGES = ["M0", "M1", "M2", "M3"]
# The alt branch's OWN sequential trajectory (Dolly-initialized), mirrors
# SEQUENTIAL_STAGES exactly - M0 is shared (same base model either way).
ALT_SEQUENTIAL_STAGES = ["M0", "M1_alt", "M2_alt", "M3_alt"]
# The genuinely NEW comparison the alt branch exists to answer: does each
# original stage's direction match its alt-branch (different M1 data)
# counterpart? High similarity -> the representational finding at that
# stage is not dataset-specific. Low similarity -> it is.
CROSS_BRANCH_PAIRS = [
    ("M1", "M1_alt"),
    ("M2", "M2_alt"),
    ("M3", "M3_alt"),
    ("M3_direct", "M3_direct_alt"),
]
ACT_DIR = Path("results/activations")
OUT_DIR = Path("results/refusal_direction")
POS_QUADRANT = "A"
NEG_QUADRANT = "D"


def activations_available(stage, act_dir=None):
    """Whether stage's activations have actually been extracted yet - the
    alt branch trains and pushes independently across sessions (Drive-space
    constrained), so at any given time some of STAGES may not be ready.
    Checked by file existence, not a live HF call (unlike
    try_load_stage_model in src/training/model.py, used by the two scripts
    that generate activations from the model in the first place) - this
    just reads whatever eval_extract_activations.py already produced."""
    act_dir = act_dir or ACT_DIR
    return (act_dir / f"{stage}_pooled.npy").exists() and (act_dir / f"{stage}_metadata.json").exists()


def load_stage(stage):
    pooled = np.load(ACT_DIR / f"{stage}_pooled.npy")  # (n_prompts, n_layers, hidden_dim)
    with open(ACT_DIR / f"{stage}_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    quadrants = np.array([row["quadrant"] for row in meta])
    splits = np.array([row.get("split") or "" for row in meta])
    return pooled, quadrants, splits


def filter_to_direction_estimation_split(pooled, quadrants, splits):
    """Keep all quadrant B/C rows (split is always "" for them - irrelevant
    to the direction) plus only the "direction_estimation" half of quadrant
    A/D. Apply this before diff_in_means_direction / per_layer_separability
    so the direction is never estimated on the same A/D prompts causal
    ablation/steering later tests its effect on - see
    build_eval_set.assign_direction_split's docstring for why that matters.
    Safe no-op on activations extracted before the split existed (splits
    all ""): those rows keep quadrant B/C, but any A/D row there has no
    split recorded and would be silently dropped here - re-extract first."""
    keep = (quadrants != "A") & (quadrants != "D") | (splits == "direction_estimation")
    return pooled[keep], quadrants[keep]


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

    available_stages = [s for s in STAGES if activations_available(s)]
    skipped_stages = [s for s in STAGES if s not in available_stages]
    if skipped_stages:
        print(f"Skipping (activations not yet extracted): {skipped_stages}")
    if "M0" not in available_stages:
        raise RuntimeError(
            "M0's activations are required as the baseline for every comparison below and "
            "are missing - run eval_extract_activations.py for M0 first."
        )

    directions = {}
    quadrants_by_stage = {}
    pooled_by_stage = {}

    for stage in available_stages:
        pooled, quadrants, splits = load_stage(stage)
        # Direction estimated ONLY on the direction_estimation half of A/D -
        # keeps causal ablation/steering's later held_out_behavioral test
        # from being circular (see build_eval_set.assign_direction_split).
        # Everything else below (quadrant_projections, cross-branch cosine
        # sim, etc.) uses the FULL pooled/quadrants - only the direction
        # VECTOR itself needs the estimation-only restriction.
        pooled_for_direction, quadrants_for_direction = filter_to_direction_estimation_split(pooled, quadrants, splits)
        direction = diff_in_means_direction(pooled_for_direction, quadrants_for_direction)
        directions[stage] = direction
        quadrants_by_stage[stage] = quadrants
        pooled_by_stage[stage] = pooled
        np.save(OUT_DIR / f"{stage}_direction.npy", direction)
        print(f"{stage}: direction shape {direction.shape}")

    # 1. Cross-stage cosine similarity, per layer, vs M0 (the untrained baseline)
    cosine_vs_m0 = {
        stage: cosine_similarity_per_layer(directions["M0"], directions[stage]).tolist()
        for stage in available_stages
    }

    cosine_vs_m3 = {}
    if "M3" in directions:
        cosine_vs_m3 = {
            stage: cosine_similarity_per_layer(directions["M3"], directions[stage]).tolist()
            for stage in available_stages
        }

    # Also the adjacent-stage chain (M0->M1->M2->M3), often more informative than "vs M0".
    # SEQUENTIAL_STAGES only -- M3_direct is a parallel branch, not the next
    # link after M3 (see SEQUENTIAL_STAGES comment above). Only pairs where
    # BOTH stages are available are computed - partial chains are fine.
    cosine_adjacent = {}
    for a, b in zip(SEQUENTIAL_STAGES[:-1], SEQUENTIAL_STAGES[1:]):
        if a in directions and b in directions:
            cosine_adjacent[f"{a}_vs_{b}"] = cosine_similarity_per_layer(directions[a], directions[b]).tolist()

    # The alt branch's OWN adjacent-stage chain (Dolly-initialized), mirrors
    # "adjacent" above exactly - lets you ask "does M0->M1_alt show the same
    # big rotation Finding 3 found for M0->M1?" independently of any
    # cross-branch comparison.
    cosine_adjacent_alt = {}
    for a, b in zip(ALT_SEQUENTIAL_STAGES[:-1], ALT_SEQUENTIAL_STAGES[1:]):
        if a in directions and b in directions:
            cosine_adjacent_alt[f"{a}_vs_{b}"] = cosine_similarity_per_layer(directions[a], directions[b]).tolist()

    # M3_direct's own two meaningful comparisons (PROJECT_CONTEXT.md/
    # HANDOFF.md M3_direct notes): does direct DPO from M1 rotate by a
    # similar amount to M1->M2's safety-SFT step (M1_vs_M3_direct, not
    # previously computed anywhere), and do the two DPO endpoints converge
    # (M3_direct_vs_M3 -- same values as vs_M3["M3_direct"], repeated here
    # under an explicit, unambiguous label rather than relying on a reader
    # to find it nested under "M3" as the reference stage). Same pattern for
    # the alt branch's M3_direct_alt.
    cosine_direct_branch = {}
    if "M1" in directions and "M3_direct" in directions:
        cosine_direct_branch["M1_vs_M3_direct"] = cosine_similarity_per_layer(
            directions["M1"], directions["M3_direct"]
        ).tolist()
    if "M3_direct" in directions and "M3" in directions:
        cosine_direct_branch["M3_direct_vs_M3"] = cosine_similarity_per_layer(
            directions["M3_direct"], directions["M3"]
        ).tolist()
    if "M1_alt" in directions and "M3_direct_alt" in directions:
        cosine_direct_branch["M1_alt_vs_M3_direct_alt"] = cosine_similarity_per_layer(
            directions["M1_alt"], directions["M3_direct_alt"]
        ).tolist()
    if "M3_direct_alt" in directions and "M3_alt" in directions:
        cosine_direct_branch["M3_direct_alt_vs_M3_alt"] = cosine_similarity_per_layer(
            directions["M3_direct_alt"], directions["M3_alt"]
        ).tolist()

    # Cross-branch: does each original stage's direction match its
    # alt-branch (different M1 data) counterpart's direction? This is the
    # comparison the whole alt branch exists to enable - high similarity
    # here means a given finding is NOT dataset-specific; low similarity
    # means it is. Computed per available pair, so this fills in
    # incrementally as more alt-branch stages finish training.
    cosine_cross_branch = {}
    for orig, alt in CROSS_BRANCH_PAIRS:
        if orig in directions and alt in directions:
            cosine_cross_branch[f"{orig}_vs_{alt}"] = cosine_similarity_per_layer(
                directions[orig], directions[alt]
            ).tolist()

    from src.io_utils import write_json

    write_json(
        {
            "vs_M0": cosine_vs_m0,
            "vs_M3": cosine_vs_m3,
            "adjacent": cosine_adjacent,
            "adjacent_alt": cosine_adjacent_alt,
            "direct_branch": cosine_direct_branch,
            "cross_branch": cosine_cross_branch,
        },
        OUT_DIR / "cosine_similarity.json",
    )

    # 2. Mean projection per quadrant per stage, per layer (each stage projected onto ITS OWN direction)
    projections = {
        stage: mean_projection_by_quadrant(pooled_by_stage[stage], quadrants_by_stage[stage], directions[stage])
        for stage in available_stages
    }

    write_json(projections, OUT_DIR / "quadrant_projections.json")

    print(f"\nSaved directions, cosine_similarity.json, quadrant_projections.json to {OUT_DIR}/")
    if cosine_cross_branch:
        print(f"Cross-branch comparisons computed: {list(cosine_cross_branch.keys())}")
    missing_cross_branch = [f"{o}_vs_{a}" for o, a in CROSS_BRANCH_PAIRS if f"{o}_vs_{a}" not in cosine_cross_branch]
    if missing_cross_branch:
        print(f"Cross-branch comparisons NOT YET available (missing activations): {missing_cross_branch}")


if __name__ == "__main__":
    main()