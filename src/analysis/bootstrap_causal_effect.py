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
    """pairs: list of (baseline_bool, other_bool). Effect = relative
    reduction in rate: (baseline_rate - other_rate) / baseline_rate."""
    rng = np.random.default_rng(seed)
    n = len(pairs)
    pairs_arr = np.array(pairs)

    effects = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = pairs_arr[idx]
        baseline_rate = sample[:, 0].mean()
        other_rate = sample[:, 1].mean()
        if baseline_rate > 0:
            effects.append((baseline_rate - other_rate) / baseline_rate)

    effects = np.array(effects)
    lo = np.percentile(effects, (1 - ci) / 2 * 100)
    hi = np.percentile(effects, (1 + ci) / 2 * 100)
    return float(np.mean(effects)), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results/raw/causal_ablation_raw_wide.json")
    parser.add_argument("--quadrant", default="C")
    parser.add_argument("--category", default="soft_deflection")
    args = parser.parse_args()

    rows = load_json(args.file)
    pairs = build_paired_outcomes(rows, args.quadrant, args.category)
    print(f"Paired on {len(pairs)} prompts (quadrant {args.quadrant}, category={args.category}).")

    mean_effect, lo, hi = bootstrap_effect_ci(pairs)
    print(f"Relative reduction: {mean_effect:.1%} [{lo:.1%}, {hi:.1%}] (95% bootstrap CI, {N_BOOTSTRAP} resamples)")

    out_path = Path(f"results/summaries/bootstrap_ci_{Path(args.file).stem}_{args.quadrant}_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_pairs": len(pairs), "mean_relative_reduction": mean_effect, "ci_low": lo, "ci_high": hi}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()