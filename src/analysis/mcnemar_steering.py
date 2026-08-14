"""
Paired McNemar exact test on the steering result: does the refusal rate on
quadrant D genuinely shift between M3_baseline and M3_steered, matched by
prompt (not just eyeballed via CI overlap)?
"""
import argparse
from collections import defaultdict

from statsmodels.stats.contingency_tables import mcnemar

from src.analysis.summarize_causal_ablation import classify_completion
from src.io_utils import load_json


def build_contingency(rows, category="refusal"):
    by_prompt = defaultdict(dict)
    for row in rows:
        by_prompt[row["prompt"]][row["stage"]] = classify_completion(row["response"]) == category

    both = [v for v in by_prompt.values() if "M3_baseline" in v and "M3_steered" in v]
    yes_yes = sum(1 for v in both if v["M3_baseline"] and v["M3_steered"])
    yes_no = sum(1 for v in both if v["M3_baseline"] and not v["M3_steered"])
    no_yes = sum(1 for v in both if not v["M3_baseline"] and v["M3_steered"])
    no_no = sum(1 for v in both if not v["M3_baseline"] and not v["M3_steered"])
    return [[yes_yes, yes_no], [no_yes, no_no]], len(both)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results/raw/steering_raw_D_L21.json")
    parser.add_argument("--category", default="refusal")
    args = parser.parse_args()

    rows = load_json(args.file)
    table, n_paired = build_contingency(rows, args.category)
    print(f"Paired on {n_paired} prompts. Contingency table (baseline x steered), category={args.category}:")
    print(f"  yes->yes: {table[0][0]}   yes->no: {table[0][1]}")
    print(f"  no->yes:  {table[1][0]}   no->no:  {table[1][1]}")

    result = mcnemar(table, exact=True)
    print(f"McNemar exact p-value: {result.pvalue:.6f}")


if __name__ == "__main__":
    main()