"""
Component 5 refinement: paired McNemar's test on causal_ablation_raw.json.
The 370 prompts are the SAME across M3_baseline/M3_ablated (paired design),
so per-prompt comparison is the correct test, not two independent Wilson
CIs. Reuses classify_completion() from summarize_causal_ablation.py --
does not reimplement classification.
"""
import json
from collections import defaultdict

from statsmodels.stats.contingency_tables import mcnemar

from src.summarize_causal_ablation import classify_completion, load_rows

QUADRANTS_TO_TEST = {
    "C": "soft_deflection",  # headline: does ablation reduce C's soft-deflection?
    "A": "refusal",          # selectivity check: does it also wipe out A's legitimate refusal?
}


def build_paired_outcomes(rows, quadrant, target_category):
    by_prompt = defaultdict(dict)
    for row in rows:
        if row["quadrant"] != quadrant:
            continue
        cat = classify_completion(row["response"])
        by_prompt[row["prompt"]][row["stage"]] = (cat == target_category)
    pairs = []
    for prompt, conditions in by_prompt.items():
        if "M3_baseline" in conditions and "M3_ablated" in conditions:
            pairs.append((conditions["M3_baseline"], conditions["M3_ablated"]))
    return pairs


def contingency_table(pairs):
    yes_yes = sum(1 for b, a in pairs if b and a)
    yes_no = sum(1 for b, a in pairs if b and not a)
    no_yes = sum(1 for b, a in pairs if not b and a)
    no_no = sum(1 for b, a in pairs if not b and not a)
    return [[yes_yes, yes_no], [no_yes, no_no]]


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} rows.\n")
    for quadrant, category in QUADRANTS_TO_TEST.items():
        pairs = build_paired_outcomes(rows, quadrant, category)
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