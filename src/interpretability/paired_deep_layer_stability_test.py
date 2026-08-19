"""
Formal paired comparison of deep-layer bootstrap stability between the
direct-DPO branches (M3_direct, M3_direct_alt) and their M2-mediated
counterparts (M3, M3_alt) - turns README Finding 3 / Open Questions'
"does the deep-layer stability difference actually survive a formal paired
comparison, or is some of it noise from comparing two descriptive ranges?"
into an actual tested claim, per CLAUDE.md Next-step #3.

Why this can be a PAIRED test, not just an unpaired distributional one:
bootstrap_direction_stability.py calls bootstrap_directions(pooled,
quadrants, seed=SEED) with the SAME constant SEED for every stage, and
every stage scores the identical, fixed, identically-ordered 370-prompt
eval set (CLAUDE.md core design #5 - same per-quadrant counts everywhere).
That means replicate i's resampled quadrant-A/D PROMPT POSITIONS are
IDENTICAL in composition across every stage - only the underlying model
activations differ. So replicate i's deep-layer stability value for M3 and
replicate i's for M3_direct are directly comparable pairs (same resampled
prompt subset, different model), and a Wilcoxon signed-rank test - which
needs matched pairs, not just two independent samples - is the right,
higher-powered tool here (vs. an unpaired Mann-Whitney U, which would
throw away that shared structure).

Requires bootstrap_direction_stability.py to have been re-run with the
current code (persists raw per-replicate `raw_sims` for DEEP_LAYERS) -
older output files that predate that change won't have it; this script
fails with a clear message rather than silently comparing nothing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.interpretability.bootstrap_direction_stability import DEEP_LAYERS

# Each pair: (direct-DPO stage, its M2-mediated counterpart on the SAME
# branch/dataset). Comparing within a branch keeps "same model family,
# same M1 data" fixed and isolates the one thing that differs: whether DPO
# went through safety-SFT (M2) first or was applied directly to M1.
DIRECT_VS_MEDIATED_PAIRS = [
    ("M3_direct", "M3"),
    ("M3_direct_alt", "M3_alt"),
]
IN_PATH = Path("results/interpretability/bootstrap_direction_stability.json")
OUT_PATH = Path("results/interpretability/paired_deep_layer_stability_test.json")


def deep_layer_mean_sims(stage_data, layers=None):
    """stage_data: bootstrap_direction_stability.json's out[stage] dict
    (keys are layer indices, as strings once loaded from JSON). Returns a
    (n_bootstrap,) array: for each replicate, the mean bootstrap-vs-original
    cosine similarity across `layers` (default DEEP_LAYERS) - the same
    "how stable is the deep-layer direction" quantity README Finding 3
    describes as a 16-28 band, reduced to one number per replicate."""
    layers = layers if layers is not None else DEEP_LAYERS
    missing_raw = [l for l in layers if str(l) in stage_data and "raw_sims" not in stage_data[str(l)]]
    if missing_raw:
        raise ValueError(
            f"raw_sims missing for layer(s) {missing_raw} - re-run "
            "bootstrap_direction_stability.py with the current code (persists "
            "raw per-replicate sims for DEEP_LAYERS) before running this test."
        )
    available_layers = [l for l in layers if str(l) in stage_data]
    if not available_layers:
        raise ValueError(f"None of the requested layers {layers} are present in this stage's data.")
    raw = np.array([stage_data[str(l)]["raw_sims"] for l in available_layers])  # (n_layers, n_bootstrap)
    return raw.mean(axis=0)  # (n_bootstrap,)


def paired_stability_test(direct_sims, mediated_sims, ci=0.95):
    """Wilcoxon signed-rank test, paired by replicate index (see module
    docstring for why that pairing is valid). direct_sims, mediated_sims:
    (n_bootstrap,) arrays from deep_layer_mean_sims, SAME length, index i
    in both corresponds to the same resampled prompt subset."""
    if len(direct_sims) != len(mediated_sims):
        raise ValueError(
            f"direct_sims and mediated_sims must have the same length (same "
            f"N_BOOTSTRAP) to be paired by replicate index - got "
            f"{len(direct_sims)} vs {len(mediated_sims)}."
        )
    diff = direct_sims - mediated_sims  # positive -> direct-DPO more stable
    stat, p_value = wilcoxon(direct_sims, mediated_sims)

    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "n_bootstrap_replicates": int(len(diff)),
        "direct_mean": float(direct_sims.mean()),
        "mediated_mean": float(mediated_sims.mean()),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
        "difference_direct_minus_mediated": {
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "ci_low_2.5pct": float(np.percentile(diff, lo_pct)),
            "ci_high_97.5pct": float(np.percentile(diff, hi_pct)),
        },
        "frac_replicates_direct_gt_mediated": float((diff > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(IN_PATH))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    in_path = Path(args.file)
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found - run bootstrap_direction_stability.py first."
        )
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    out = {"per_branch": {}}
    per_branch_diffs = []
    for direct_stage, mediated_stage in DIRECT_VS_MEDIATED_PAIRS:
        if direct_stage not in data or mediated_stage not in data:
            missing = [s for s in (direct_stage, mediated_stage) if s not in data]
            print(f"{direct_stage} vs {mediated_stage}: SKIPPED, missing stage(s) {missing}")
            continue
        direct_sims = deep_layer_mean_sims(data[direct_stage])
        mediated_sims = deep_layer_mean_sims(data[mediated_stage])
        result = paired_stability_test(direct_sims, mediated_sims, ci=1 - args.alpha)
        out["per_branch"][f"{direct_stage}_vs_{mediated_stage}"] = result
        per_branch_diffs.append(direct_sims - mediated_sims)

        sig = "SIGNIFICANT" if result["p_value"] < args.alpha else "not significant"
        print(f"{direct_stage} vs {mediated_stage} (deep layers {DEEP_LAYERS[0]}-{DEEP_LAYERS[-1]}):")
        print(f"  direct mean {result['direct_mean']:.4f} vs mediated mean {result['mediated_mean']:.4f}, "
              f"diff {result['difference_direct_minus_mediated']['mean']:+.4f}")
        print(f"  Wilcoxon signed-rank: statistic={result['wilcoxon_statistic']:.1f}, "
              f"p={result['p_value']:.2e}  ({sig} at alpha={args.alpha})")

    if len(per_branch_diffs) == len(DIRECT_VS_MEDIATED_PAIRS) and len(per_branch_diffs) > 1:
        # Pooled evidence across both independent branch replications - each
        # branch's diffs stay paired-by-replicate-index WITHIN itself; we're
        # just concatenating two independent paired experiments' evidence
        # for the same underlying claim (direct-DPO branches are more
        # deep-layer-stable than M2-mediated ones) into one combined test.
        pooled_diff = np.concatenate(per_branch_diffs)
        stat, p_value = wilcoxon(pooled_diff)
        out["pooled_across_branches"] = {
            "n_values": int(len(pooled_diff)),
            "mean_diff": float(pooled_diff.mean()),
            "wilcoxon_statistic": float(stat),
            "p_value": float(p_value),
        }
        sig = "SIGNIFICANT" if p_value < args.alpha else "not significant"
        print(f"\nPooled across both branches: mean diff {pooled_diff.mean():+.4f}, "
              f"Wilcoxon p={p_value:.2e} ({sig} at alpha={args.alpha})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
