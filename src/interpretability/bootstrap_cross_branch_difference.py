"""
Bootstrap the DIFFERENCE in cross-branch direction similarity between the
M2-mediated path (M2, M3 - both trained through safety-SFT) and the
direct-DPO path (M3_direct - skips safety-SFT), instead of eyeballing two
point-estimate ranges (README Finding 3, second paragraph: M2=0.919,
M3=0.916 vs M3_direct=0.875 - "I'm calling this a pattern, not a finding
yet ... I haven't tested whether it's bigger than resampling noise").

Reuses eval_refusal_direction.py's diff_in_means_direction, same as
bootstrap_direction_stability.py, but resamples TWO branches (orig, alt) at
once instead of one stage.

Resampling design: both branches of a cross-branch pair (e.g. M3 and
M3_alt) score the exact same fixed, ordered 370-prompt controlled eval set
(CLAUDE.md core design decision #5) - so quadrant-A/D PROMPT POSITIONS can
be resampled once per replicate and applied to BOTH sides jointly. This is
the same "resample the pair jointly" logic bootstrap_causal_effect.py uses
for paired outcomes, and is more powerful than resampling each side with
independent randomness because it isolates prompt-selection noise from
genuine between-branch differences. `bootstrap_cross_branch_similarity`
asserts the two metadata quadrant orders match before doing this - if a
future data change breaks that assumption, this fails loudly instead of
silently pairing mismatched prompts.

Each cross-branch PAIR's bootstrap distribution is independent of every
other pair's (different underlying activations, different RNG stream), so
combining M2's and M3's distributions into one "M2-mediated" group and
differencing against M3_direct's distribution is the standard two
bootstrapped-groups approach to a CI on a difference of means.
"""
import json
from pathlib import Path

import numpy as np

from src.analysis.eval_refusal_direction import (
    CROSS_BRANCH_PAIRS,
    activations_available,
    cosine_similarity_per_layer,
    diff_in_means_direction,
    filter_to_direction_estimation_split,
    load_stage,
)

N_BOOTSTRAP = 1000  # matches bootstrap_direction_stability.py's B=1000
SEED = 0
# The two groups README Finding 3 eyeballs against each other. M1 isn't part
# of either - it's the shared pre-safety-training, pre-DPO ancestor.
M2_MEDIATED_PAIRS = [("M2", "M2_alt"), ("M3", "M3_alt")]
DIRECT_DPO_PAIRS = [("M3_direct", "M3_direct_alt")]
OUT_PATH = Path("results/interpretability/bootstrap_cross_branch_difference.json")


def bootstrap_cross_branch_similarity(
    pooled_orig, quadrants_orig, pooled_alt, quadrants_alt,
    n_bootstrap=N_BOOTSTRAP, seed=SEED,
):
    """Returns (per_layer_sims, mean_sims_excl_layer0):
      per_layer_sims: (n_bootstrap, n_layers) cosine similarity between the
        two branches' resampled directions, per replicate, per layer.
      mean_sims_excl_layer0: (n_bootstrap,) mean over layers 1..n-1 - layer
        0 is always exactly 0.0 (template-token artifact, see
        summarize_cross_branch.py's direction_cross_branch_similarity),
        excluding it matches how the README's headline numbers are computed.
    """
    if quadrants_orig.shape != quadrants_alt.shape or not (quadrants_orig == quadrants_alt).all():
        raise ValueError(
            "orig/alt quadrant order must match - both branches must score the "
            "same fixed, identically-ordered eval set for joint resampling to "
            "be valid (see module docstring)."
        )
    rng = np.random.default_rng(seed)
    a_idx = np.where(quadrants_orig == "A")[0]
    d_idx = np.where(quadrants_orig == "D")[0]

    per_layer_sims = []
    for _ in range(n_bootstrap):
        a_sample = rng.choice(a_idx, size=len(a_idx), replace=True)
        d_sample = rng.choice(d_idx, size=len(d_idx), replace=True)
        sample_idx = np.concatenate([a_sample, d_sample])
        sample_quadrants = np.array(["A"] * len(a_sample) + ["D"] * len(d_sample))

        dir_orig = diff_in_means_direction(pooled_orig[sample_idx], sample_quadrants)
        dir_alt = diff_in_means_direction(pooled_alt[sample_idx], sample_quadrants)
        per_layer_sims.append(cosine_similarity_per_layer(dir_orig, dir_alt))

    per_layer_sims = np.stack(per_layer_sims)  # (n_bootstrap, n_layers)
    mean_sims = per_layer_sims[:, 1:].mean(axis=1) if per_layer_sims.shape[1] > 1 else per_layer_sims.mean(axis=1)
    return per_layer_sims, mean_sims


def bootstrap_group_difference(mediated_mean_arrays, direct_mean_arrays, ci=0.95):
    """mediated_mean_arrays / direct_mean_arrays: lists of (n_bootstrap,)
    arrays, one per cross-branch pair in that group. Averaging pairs within
    a group first, then differencing group means elementwise by replicate
    index, gives a percentile CI on the group-level difference."""
    mediated_group = np.stack(mediated_mean_arrays).mean(axis=0)  # (n_bootstrap,)
    direct_group = np.stack(direct_mean_arrays).mean(axis=0)
    diff = mediated_group - direct_group

    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "n_bootstrap_replicates": int(len(diff)),
        "mediated_group_mean": float(mediated_group.mean()),
        "direct_group_mean": float(direct_group.mean()),
        "difference_mediated_minus_direct": {
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "std": float(diff.std()),
            "ci_low_2.5pct": float(np.percentile(diff, lo_pct)),
            "ci_high_97.5pct": float(np.percentile(diff, hi_pct)),
        },
        "frac_replicates_mediated_gt_direct": float((diff > 0).mean()),
    }


def main():
    stage_cache = {}

    def get_stage(stage):
        if stage not in stage_cache:
            pooled, quadrants, splits = load_stage(stage)
            # Estimation-split only, same rationale as everywhere else in
            # the direction component - keeps this bootstrap consistent
            # with the direction causal ablation/steering actually test.
            stage_cache[stage] = filter_to_direction_estimation_split(pooled, quadrants, splits)
        return stage_cache[stage]

    pair_mean_sims = {}
    pair_summaries = {}
    print("Bootstrapping cross-branch direction similarity per pair "
          f"(B={N_BOOTSTRAP}, joint prompt-position resampling)\n")
    for orig, alt in CROSS_BRANCH_PAIRS:
        if not (activations_available(orig) and activations_available(alt)):
            print(f"  {orig}_vs_{alt}: SKIPPED, activations not yet extracted for both sides")
            continue
        pooled_orig, quadrants_orig = get_stage(orig)
        pooled_alt, quadrants_alt = get_stage(alt)
        _, mean_sims = bootstrap_cross_branch_similarity(pooled_orig, quadrants_orig, pooled_alt, quadrants_alt)
        pair_mean_sims[f"{orig}_vs_{alt}"] = mean_sims
        pair_summaries[f"{orig}_vs_{alt}"] = {
            "mean": float(mean_sims.mean()),
            "median": float(np.median(mean_sims)),
            "std": float(mean_sims.std()),
            "ci_low_2.5pct": float(np.percentile(mean_sims, 2.5)),
            "ci_high_97.5pct": float(np.percentile(mean_sims, 97.5)),
        }
        print(f"  {orig}_vs_{alt}: mean cross-branch similarity = {mean_sims.mean():.4f} "
              f"(95% CI [{pair_summaries[f'{orig}_vs_{alt}']['ci_low_2.5pct']:.4f}, "
              f"{pair_summaries[f'{orig}_vs_{alt}']['ci_high_97.5pct']:.4f}])")

    out = {"per_pair": pair_summaries}

    mediated_keys = [f"{o}_vs_{a}" for o, a in M2_MEDIATED_PAIRS]
    direct_keys = [f"{o}_vs_{a}" for o, a in DIRECT_DPO_PAIRS]
    have_mediated = all(k in pair_mean_sims for k in mediated_keys)
    have_direct = all(k in pair_mean_sims for k in direct_keys)

    if have_mediated and have_direct:
        group_diff = bootstrap_group_difference(
            [pair_mean_sims[k] for k in mediated_keys],
            [pair_mean_sims[k] for k in direct_keys],
        )
        out["group_comparison"] = {
            "mediated_pairs": mediated_keys,
            "direct_pairs": direct_keys,
            **group_diff,
        }
        d = group_diff["difference_mediated_minus_direct"]
        print(f"\nM2-mediated ({', '.join(mediated_keys)}) vs direct-DPO ({', '.join(direct_keys)}):")
        print(f"  difference = {d['mean']:+.4f} (95% CI [{d['ci_low_2.5pct']:+.4f}, {d['ci_high_97.5pct']:+.4f}])")
        if d["ci_low_2.5pct"] > 0:
            print("  CI excludes 0: M2-mediated cross-branch similarity is reliably higher.")
        elif d["ci_high_97.5pct"] < 0:
            print("  CI excludes 0: direct-DPO cross-branch similarity is reliably higher.")
        else:
            print("  CI includes 0: not distinguishable from resampling noise at this sample size.")
    else:
        missing = [k for k in mediated_keys + direct_keys if k not in pair_mean_sims]
        print(f"\nGroup comparison SKIPPED - missing pair(s): {missing}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
