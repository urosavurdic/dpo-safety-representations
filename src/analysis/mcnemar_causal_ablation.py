"""
Component 5 refinement: paired McNemar's test on causal_ablation_raw.json.
The 370 prompts are the SAME across M3_baseline/M3_ablated (paired design),
so per-prompt comparison is the correct test, not two independent Wilson
CIs. Reuses classify_completion() from summarize_causal_ablation.py --
does not reimplement classification.
"""
import argparse
import json
from collections import defaultdict

from statsmodels.stats.contingency_tables import mcnemar

from src.analysis.summarize_causal_ablation import classify_completion
from src.io_utils import load_json

QUADRANTS_TO_TEST = {
    "C": "soft_deflection",  # headline: does ablation reduce C's soft-deflection?
    "A": "refusal",          # selectivity check: does it also wipe out A's legitimate refusal?
}



def build_paired_outcomes(rows, quadrant, target_category, baseline_condition, intervention_condition):
    by_prompt = defaultdict(dict)

    for row in rows:
        if row["quadrant"] != quadrant:
            continue
        cat = classify_completion(row["response"])
        by_prompt[row["prompt"]][row["stage"]] = (cat == target_category)
    pairs = []
    for prompt, conditions in by_prompt.items():
        if (baseline_condition in conditions and intervention_condition in conditions):
            pairs.append(
                (
                    conditions[baseline_condition],
                    conditions[intervention_condition],
                )
            )
    return pairs


def contingency_table(pairs):
    yes_yes = sum(1 for b, a in pairs if b and a)
    yes_no = sum(1 for b, a in pairs if b and not a)
    no_yes = sum(1 for b, a in pairs if not b and a)
    no_no = sum(1 for b, a in pairs if not b and not a)
    return [[yes_yes, yes_no], [no_yes, no_no]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the causal ablation raw JSON to analyze")
    parser.add_argument("--conditions", nargs=2, default=["M3_baseline", "M3_ablated"], help="Baseline and intervention condition names")
    args = parser.parse_args()
    rows = load_json(args.file)
    baseline_condition, intervention_condition = args.conditions
    print(f"Loaded {len(rows)} rows.\n")
    for quadrant, category in QUADRANTS_TO_TEST.items():
        pairs = build_paired_outcomes(rows, quadrant, baseline_condition, intervention_condition)
        table = contingency_table(pairs)
        result = mcnemar(table, exact=True)
        print(f"=== Quadrant {quadrant}, category '{category}' (n={len(pairs)}) ===")
        print(f"  baseline=yes -> ablated=yes: {table[0][0]}")
        print(f"  baseline=yes -> ablated=no : {table[0][1]}  (switched AWAY under ablation)")
        print(f"  baseline=no  -> ablated=yes: {table[1][0]}  (switched TOWARD under ablation)")
        print(f"  baseline=no  -> ablated=no : {table[1][1]}")
        print(f"  McNemar exact p-value: {result.pvalue:.6f}\n")


if __name__ == "__main__":
    main()