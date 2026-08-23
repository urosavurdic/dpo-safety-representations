"""
Paired McNemar exact test on a steering result file: does a category's rate
genuinely shift between a baseline and a steered condition, matched by
prompt (not just eyeballed via CI overlap)?

Previously hardcoded to literal condition names "M3_baseline"/"M3_steered"
and defaulted --file to the single pre-eval_steering_v2 exploratory file
steering_raw_D_L21.json -- the exact same bug class CLAUDE.md documents as
already found and fixed in summarize_steering.py, except this one was never
actually fixed. Concretely: every eval_steering_v2.py run (any stage,
config, or quadrant set) names its conditions "{tag}_baseline"/
"{tag}_steered", e.g. "M3_L24_quadrant_a_projection_coef1_QAD_baseline" --
never the literal string "M3_baseline". Filtering on that literal string
against a real eval_steering_v2.py output file silently matched zero rows
(0 paired, not an error), which is exactly the kind of "ran without
erroring but was checking nothing" failure mode this project's testing
convention is meant to catch -- caught here specifically because a fresh
8-stage run (Next Steps item 1) would have hit this on every single stage,
not just M3.

Fixed the same way summarize_steering.py was: derive condition pairs from
the file's actual stage names (reuses summarize_steering.find_condition_pairs
rather than reimplementing it) instead of a hardcoded literal, require
--file explicitly (no default to silently fall back to), and require
--quadrant explicitly too (mirrors bootstrap_causal_effect.py's convention --
pooling quadrant A and D together under one refusal-rate test would
conflate two prompts sets with very different baseline refusal rates and
different intended questions, so there's no sane default to pick).

Still handles the old, deprecated exploratory files (steering_raw_D.json,
steering_raw_D_L21.json, kept as evidence per CLAUDE.md's steering
methodology history) correctly -- find_condition_pairs works off the
"_baseline"/"_steered" suffix convention those files also happen to use
(their stage names literally are "M3_baseline"/"M3_steered"), so nothing
about this fix requires migrating them.
"""
import argparse
from collections import defaultdict

from statsmodels.stats.contingency_tables import mcnemar

from src.analysis.summarize_causal_ablation import classify_completion
from src.analysis.summarize_steering import find_condition_pairs
from src.io_utils import load_json


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
                         help="Raw steering results file, e.g. results/raw/steering_v2_M3_L24_...json. "
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