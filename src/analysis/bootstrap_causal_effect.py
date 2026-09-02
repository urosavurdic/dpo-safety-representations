"""
Bootstrap CI on the causal ablation effect SIZE (not just its direction/
significance -- McNemar already handles that correctly for n=20). Answers:
given quadrant C's small n, how precisely is the MAGNITUDE of the effect
estimated?
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.analysis.summarize_causal_ablation import classify_completion
from src.io_utils import load_json
from src.v2_binding_guard import add_binding_cli_args, load_guarded_raw

N_BOOTSTRAP = 2000
SEED = 0


def build_paired_outcomes(rows, quadrant, category):
    """Returns list of (baseline_bool, other_bool) pairs, one per prompt
    with both a *_baseline and a non-baseline condition present, for the
    given quadrant/category. Works for both causal-ablation files
    (M3_baseline/M3_ablated) and steering files (M3_baseline/M3_steered)
    without hardcoding either pair of names."""
    by_prompt = defaultdict(dict)
    for row in rows:
        if row["quadrant"] != quadrant:
            continue
        is_cat = classify_completion(row["response"]) == category
        by_prompt[row["prompt"]][row["stage"]] = is_cat

    pairs = []
    for stage_map in by_prompt.values():
        base_key = next((k for k in stage_map if "baseline" in k), None)
        other_key = next((k for k in stage_map if k != base_key), None)
        if base_key is not None and other_key is not None:
            pairs.append((stage_map[base_key], stage_map[other_key]))
    return pairs


def bootstrap_effect_ci(pairs, n_bootstrap=N_BOOTSTRAP, seed=SEED, ci=0.95):
    """pairs: list of (baseline_bool, other_bool). Reports both the absolute
    effect (baseline_rate - other_rate) and the relative effect
    ((baseline_rate - other_rate) / baseline_rate) with 95% bootstrap CIs on
    each, per PROJECT_CONTEXT.md's bootstrap-causal-effect spec (section 24:
    "absolute effect, relative effect where meaningful"). Resamples across
    prompt PAIRS jointly (never independently resamples the two conditions)."""
    rng = np.random.default_rng(seed)
    n = len(pairs)
    pairs_arr = np.array(pairs)

    abs_effects = []
    rel_effects = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = pairs_arr[idx]
        baseline_rate = sample[:, 0].mean()
        other_rate = sample[:, 1].mean()
        abs_effects.append(baseline_rate - other_rate)
        if baseline_rate > 0:
            rel_effects.append((baseline_rate - other_rate) / baseline_rate)
        # else: relative effect undefined for this replicate (0/0), skipped -
        # only affects rel_effects' effective n, not abs_effects'.

    abs_effects = np.array(abs_effects)
    rel_effects = np.array(rel_effects)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "n_bootstrap_replicates": n_bootstrap,
        "absolute_effect": {
            "mean": float(np.mean(abs_effects)),
            "ci_low": float(np.percentile(abs_effects, lo_pct)),
            "ci_high": float(np.percentile(abs_effects, hi_pct)),
        },
        "relative_effect": {
            "mean": float(np.mean(rel_effects)) if len(rel_effects) else None,
            "ci_low": float(np.percentile(rel_effects, lo_pct)) if len(rel_effects) else None,
            "ci_high": float(np.percentile(rel_effects, hi_pct)) if len(rel_effects) else None,
            "n_defined_replicates": int(len(rel_effects)),  # < n_bootstrap iff some
            # resamples had baseline_rate==0 (relative effect undefined, 0/0)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results/raw/causal_ablation_v2_M3_L24-28.json")
    parser.add_argument("--quadrant", default="C")
    parser.add_argument("--category", default="soft_deflection")
    add_binding_cli_args(parser)
    args = parser.parse_args()

    rows = load_guarded_raw(
        args.file,
        benchmark_sha256=args.expect_benchmark_sha256,
        allow_unbound=args.allow_unbound,
    )
    pairs = build_paired_outcomes(rows, args.quadrant, args.category)
    print(f"Paired on {len(pairs)} prompts (quadrant {args.quadrant}, category={args.category}).")

    result = bootstrap_effect_ci(pairs)
    abs_e, rel_e = result["absolute_effect"], result["relative_effect"]
    print(f"Absolute reduction: {abs_e['mean']:.3f} [{abs_e['ci_low']:.3f}, {abs_e['ci_high']:.3f}] "
          f"(95% bootstrap CI, {N_BOOTSTRAP} resamples)")
    if rel_e["mean"] is not None:
        print(f"Relative reduction: {rel_e['mean']:.1%} [{rel_e['ci_low']:.1%}, {rel_e['ci_high']:.1%}] "
              f"(95% bootstrap CI, {rel_e['n_defined_replicates']}/{N_BOOTSTRAP} resamples with baseline>0)")
    else:
        print("Relative reduction: undefined (baseline rate was 0 in every resample)")

    out_path = Path(f"results/summaries/bootstrap_ci_{Path(args.file).stem}_{args.quadrant}_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_pairs": len(pairs), **result}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()