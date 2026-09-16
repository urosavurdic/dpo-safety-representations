"""Paired McNemar exact test on a steering result file.

Asks whether a category's rate genuinely shifts between a baseline and a
steered condition, matched by prompt rather than judged by CI overlap.

--file and --quadrant are both required. Pooling quadrants A and D under one
refusal-rate test would conflate prompt sets with different baseline rates and
different questions, so there is no sane default to pick. Condition pairs are
derived from the file's own names via summarize_steering.find_condition_pairs,
which keys off the "_baseline"/"_steered" suffix convention -- real run files
name conditions like "M3_L24_..._coef1_QAD_baseline", never a bare
"M3_baseline".
"""
import argparse
from collections import defaultdict

from statsmodels.stats.contingency_tables import mcnemar

from src.common.refusal_classifier import classify_completion
from src.analysis.summarize_steering import find_condition_pairs
from src.common.io import load_json


def build_contingency(rows, baseline_stage, steered_stage, category="refusal"):
    by_prompt = defaultdict(dict)
    for row in rows:
        if row["stage"] not in (baseline_stage, steered_stage):
            continue
        by_prompt[row["prompt"]][row["stage"]] = classify_completion(row["response"]) == category

    both = [v for v in by_prompt.values() if baseline_stage in v and steered_stage in v]
    yes_yes = sum(1 for v in both if v[baseline_stage] and v[steered_stage])
    yes_no = sum(1 for v in both if v[baseline_stage] and not v[steered_stage])
    no_yes = sum(1 for v in both if not v[baseline_stage] and v[steered_stage])
    no_no = sum(1 for v in both if not v[baseline_stage] and not v[steered_stage])
    return [[yes_yes, yes_no], [no_yes, no_no]], len(both)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True,
                         help="Raw steering results file, e.g. results/raw/steering_654_M3_L24_...json. "
                              "No default on purpose -- always say explicitly which file you mean.")
    parser.add_argument("--quadrant", required=True, choices=["A", "B", "C", "D"],
                         help="Which quadrant to test -- required, no default, since pooling quadrants "
                              "with different baseline rates under one test would be misleading.")
    parser.add_argument("--category", default="refusal")
    args = parser.parse_args()

    rows = load_json(args.file)
    rows = [r for r in rows if r["quadrant"] == args.quadrant]
    print(f"Loaded {len(rows)} rows for quadrant {args.quadrant} from {args.file}.")

    pairs = find_condition_pairs(rows)
    if not pairs:
        print("No baseline/steered condition pairs found for this quadrant "
              "(no stage name ends in '_baseline' with a matching '_steered' counterpart). "
              "Nothing to test.")
        return

    for baseline_stage, steered_stage in pairs:
        table, n_paired = build_contingency(rows, baseline_stage, steered_stage, args.category)
        print(f"\n=== {baseline_stage} vs {steered_stage}, category={args.category} ===")
        print(f"Paired on {n_paired} prompts. Contingency table (baseline x steered):")
        print(f"  yes->yes: {table[0][0]}   yes->no: {table[0][1]}")
        print(f"  no->yes:  {table[1][0]}   no->no:  {table[1][1]}")
        if n_paired == 0:
            print("  0 paired prompts -- skipping the test (nothing to compute a p-value from).")
            continue
        result = mcnemar(table, exact=True)
        print(f"McNemar exact p-value: {result.pvalue:.6f}")


if __name__ == "__main__":
    main()