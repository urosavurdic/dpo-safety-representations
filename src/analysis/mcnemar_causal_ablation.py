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
from src.v2_binding_guard import add_binding_cli_args, load_guarded_raw

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
    parser.add_argument("--conditions", nargs=2, default=["M3_baseline", "M3_ablated"],
                         help="Baseline and intervention condition names")
    parser.add_argument("--quadrant", default=None,
                         help="Test a single (quadrant, category) pair instead of the default "
                              "ablation-specific QUADRANTS_TO_TEST set (e.g. for steering: "
                              "--quadrant D --category refusal tests whether steering induces "
                              "refusal on benign prompts, the opposite direction from ablation).")
    parser.add_argument("--category", default=None,
                         help="Category to test on --quadrant (required if --quadrant is given).")
    add_binding_cli_args(parser)
    args = parser.parse_args()
    rows = load_guarded_raw(
        args.file,
        benchmark_sha256=args.expect_benchmark_sha256,
        allow_unbound=args.allow_unbound,
    )
    baseline_condition, intervention_condition = args.conditions
    print(f"Loaded {len(rows)} rows.\n")

    if args.quadrant is not None:
        if args.category is None:
            parser.error("--category is required when --quadrant is given")
        quadrants_to_test = {args.quadrant: args.category}
    else:
        quadrants_to_test = QUADRANTS_TO_TEST

    for quadrant, category in quadrants_to_test.items():
        pairs = build_paired_outcomes(rows, quadrant, category, baseline_condition, intervention_condition)
        table = contingency_table(pairs)
        result = mcnemar(table, exact=True)
        print(f"=== Quadrant {quadrant}, category '{category}' (n={len(pairs)}) ===")
        print(f"  {baseline_condition}=yes -> {intervention_condition}=yes: {table[0][0]}")
        print(f"  {baseline_condition}=yes -> {intervention_condition}=no : {table[0][1]}  (switched AWAY under intervention)")
        print(f"  {baseline_condition}=no  -> {intervention_condition}=yes: {table[1][0]}  (switched TOWARD under intervention)")
        print(f"  {baseline_condition}=no  -> {intervention_condition}=no : {table[1][1]}")
        print(f"  McNemar exact p-value: {result.pvalue:.6f}\n")


if __name__ == "__main__":
    main()