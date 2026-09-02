"""Deep-layer direction-stability difference: direct-DPO vs M2-mediated (WP-Stat).

**Changed for the frozen plan (analysis_plan.md, §2 "Secondary", correction #13
history):** the previous version paired the two stages by *bootstrap replicate
index* and ran a Wilcoxon signed-rank test on those pairs. That "pairs two
descriptive ranges" (README Open Question #1) and was walked back. This module
now uses a **prompt-level joint bootstrap**: within each replicate the SAME
resampled quadrant-A/D prompt positions are applied to BOTH stages of a branch
pair, each stage's direction is re-estimated on that resample, its deep-layer
stability (cosine vs that stage's own full-data direction) is computed, and the
per-replicate difference ``direct - mediated`` yields a percentile CI. This
isolates prompt-selection noise from a genuine between-stage difference, the
same design as ``bootstrap_cross_branch_difference.py``.

The old ``paired_stability_test`` (Wilcoxon) and ``deep_layer_mean_sims``
(reads pre-aggregated ``raw_sims`` from an existing JSON) are retained, marked
DEPRECATED, and are no longer used by ``main()``.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from src.analysis.eval_refusal_direction import (
    activations_available,
    cosine_similarity_per_layer,
    diff_in_means_direction,
    filter_to_direction_estimation_split,
    load_stage,
)
from src.eval_stats import BOOTSTRAP_SEED, percentile_ci
from src.interpretability.bootstrap_direction_stability import DEEP_LAYERS

DIRECT_VS_MEDIATED_PAIRS = [
    ("M3_direct", "M3"),
    ("M3_direct_alt", "M3_alt"),
]
N_BOOTSTRAP = 1000
OUT_PATH = Path("results/interpretability/paired_deep_layer_stability_test.json")


# --------------------------------------------------------------------------- #
# frozen: prompt-level joint bootstrap
# --------------------------------------------------------------------------- #
def _deep_layer_stability(pooled_sample, sample_quadrants, original_direction, layers):
    # pooled_sample is already restricted to the direction_estimation split by
    # the caller; this just re-estimates the direction on the resampled rows.
    d = diff_in_means_direction(pooled_sample, sample_quadrants)
    sims = cosine_similarity_per_layer(d, original_direction)  # (n_layers,)
    return float(np.mean([sims[l] for l in layers if l < len(sims)]))


def joint_bootstrap_deep_layer_stability_difference(
    pooled_direct, quadrants_direct,
    pooled_mediated, quadrants_mediated,
    *, layers=None, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
):
    """Per replicate: resample A/D positions jointly, apply to both stages,
    compute each stage's deep-layer stability vs its own full-data direction,
    take direct - mediated. Returns the per-replicate difference array +
    per-stage stability arrays."""
    if quadrants_direct.shape != quadrants_mediated.shape or not np.array_equal(
        quadrants_direct, quadrants_mediated
    ):
        raise ValueError(
            "direct/mediated quadrant order must match - both stages score the "
            "same fixed, identically-ordered eval set for joint resampling."
        )
    layers = list(DEEP_LAYERS if layers is None else layers)
    orig_direct = diff_in_means_direction(pooled_direct, quadrants_direct)
    orig_mediated = diff_in_means_direction(pooled_mediated, quadrants_mediated)

    a_idx = np.flatnonzero(quadrants_direct == "A")
    d_idx = np.flatnonzero(quadrants_direct == "D")
    rng = np.random.default_rng(seed)

    direct_stab = np.empty(n_bootstrap)
    mediated_stab = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        a_s = rng.choice(a_idx, size=len(a_idx), replace=True)
        d_s = rng.choice(d_idx, size=len(d_idx), replace=True)
        sample_idx = np.concatenate([a_s, d_s])
        sq = np.array(["A"] * len(a_s) + ["D"] * len(d_s))
        direct_stab[i] = _deep_layer_stability(pooled_direct[sample_idx], sq, orig_direct, layers)
        mediated_stab[i] = _deep_layer_stability(pooled_mediated[sample_idx], sq, orig_mediated, layers)

    diff = direct_stab - mediated_stab
    return {
        "layers": layers,
        "n_bootstrap": int(n_bootstrap),
        "seed": seed,
        "interval": "percentile",
        "direct_stability": percentile_ci(direct_stab),
        "mediated_stability": percentile_ci(mediated_stab),
        "difference_direct_minus_mediated": percentile_ci(diff),
        "frac_replicates_direct_gt_mediated": float((diff > 0).mean()),
        "_diff_samples": diff,
    }


# --------------------------------------------------------------------------- #
# DEPRECATED replicate-index Wilcoxon path (kept importable, not used)
# --------------------------------------------------------------------------- #
def deep_layer_mean_sims(stage_data, layers=None):  # pragma: no cover - deprecated
    warnings.warn(
        "deep_layer_mean_sims / the replicate-index Wilcoxon path is deprecated "
        "(analysis_plan.md correction #13). Use "
        "joint_bootstrap_deep_layer_stability_difference on activations.",
        DeprecationWarning, stacklevel=2,
    )
    layers = layers if layers is not None else DEEP_LAYERS
    available = [l for l in layers if str(l) in stage_data]
    raw = np.array([stage_data[str(l)]["raw_sims"] for l in available])
    return raw.mean(axis=0)


def paired_stability_test(direct_sims, mediated_sims, ci=0.95):  # pragma: no cover
    warnings.warn(
        "paired_stability_test (Wilcoxon paired by replicate index) is "
        "deprecated - see module docstring.", DeprecationWarning, stacklevel=2,
    )
    from scipy.stats import wilcoxon

    diff = np.asarray(direct_sims) - np.asarray(mediated_sims)
    stat, p_value = wilcoxon(direct_sims, mediated_sims)
    return {
        "wilcoxon_statistic": float(stat), "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "frac_replicates_direct_gt_mediated": float((diff > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    cache = {}

    def get(stage):
        if stage not in cache:
            pooled, quadrants, splits = load_stage(stage)
            cache[stage] = filter_to_direction_estimation_split(pooled, quadrants, splits)
        return cache[stage]

    out = {"method": "prompt_level_joint_bootstrap", "per_branch": {}}
    diff_samples = []
    for direct_stage, mediated_stage in DIRECT_VS_MEDIATED_PAIRS:
        if not (activations_available(direct_stage) and activations_available(mediated_stage)):
            missing = [s for s in (direct_stage, mediated_stage) if not activations_available(s)]
            print(f"{direct_stage} vs {mediated_stage}: SKIPPED, missing activations {missing}")
            continue
        p_d, q_d = get(direct_stage)
        p_m, q_m = get(mediated_stage)
        res = joint_bootstrap_deep_layer_stability_difference(
            p_d, q_d, p_m, q_m, n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        diff_samples.append(res.pop("_diff_samples"))
        out["per_branch"][f"{direct_stage}_vs_{mediated_stage}"] = res
        d = res["difference_direct_minus_mediated"]
        print(f"{direct_stage} vs {mediated_stage} (deep layers {DEEP_LAYERS[0]}-{DEEP_LAYERS[-1]}):")
        print(f"  direct {res['direct_stability']['mean']:.4f} vs mediated "
              f"{res['mediated_stability']['mean']:.4f}; diff {d['mean']:+.4f} "
              f"(95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}])")

    if len(diff_samples) == len(DIRECT_VS_MEDIATED_PAIRS) and diff_samples:
        pooled_diff = np.concatenate(diff_samples)
        out["pooled_across_branches"] = percentile_ci(pooled_diff)
        pd = out["pooled_across_branches"]
        print(f"\nPooled: mean diff {pd['mean']:+.4f} (95% CI [{pd['ci_low']:+.4f}, {pd['ci_high']:+.4f}])")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
