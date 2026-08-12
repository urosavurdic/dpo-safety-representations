"""
Component 5 summary: classify results/causal_ablation_raw.json (M3_baseline
vs M3_ablated) using the already-validated classifier functions from
eval_refusal_classifier.py, report Wilson CIs per quadrant, mirroring
Component 1's reporting style.

ASSUMPTION, NOT YET VERIFIED against eval_behavioral.py's actual call
order: precedence is degenerate -> hard refusal -> soft deflection ->
comply (inferred from decision #18's "non-degenerate-only rates reported
separately" framing). If eval_behavioral.py combines these differently,
fix classify_completion() below to match exactly -- this must mirror the
already human-validated logic, not reinvent it.
"""
import json
from collections import defaultdict
from pathlib import Path

from src.eval_refusal_classifier import classify_refusal, is_degenerate, is_soft_deflection
from src.eval_stats import rate_with_ci

CONDITIONS = ["M3_baseline", "M3_ablated"]
QUADRANTS = ["A", "B", "C", "D"]
CATEGORIES = ["degenerate", "refusal", "soft_deflection", "comply"]


def classify_completion(text):
    """ASSUMPTION -- see module docstring."""
    if is_degenerate(text):
        return "degenerate"
    if classify_refusal(text):
        return "refusal"
    if is_soft_deflection(text):
        return "soft_deflection"
    return "comply"


def load_rows(path="results/causal_ablation_raw.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} rows.")

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals = defaultdict(lambda: defaultdict(int))

    for row in rows:
        stage, quadrant = row["stage"], row["quadrant"]
        category = classify_completion(row["response"])
        counts[stage][quadrant][category] += 1
        totals[stage][quadrant] += 1

    for condition in CONDITIONS:
        print(f"\n=== {condition} ===")
        for quadrant in QUADRANTS:
            total = totals[condition][quadrant]
            if total == 0:
                print(f"  quadrant {quadrant}: NO ROWS (check condition/quadrant naming)")
                continue
            print(f"  quadrant {quadrant} (n={total}):")
            for category in CATEGORIES:
                n = counts[condition][quadrant][category]
                ci_result = rate_with_ci(n, total)
                rate = ci_result["rate"]
                lo = ci_result["ci_low"]
                hi = ci_result["ci_high"]
                print(
                    f"    {category:16s} {rate:.3f} "
                    f"[{lo:.3f}, {hi:.3f}]  ({n}/{total})"
                )

    print("\n=== Headline: quadrant C soft-deflection, baseline vs ablated ===")
    for condition in CONDITIONS:
        total = totals[condition]["C"]
        n = counts[condition]["C"]["soft_deflection"]
        if total:
            ci_result = rate_with_ci(n, total)
            rate = ci_result["rate"]
            lo = ci_result["ci_low"]
            hi = ci_result["ci_high"]
            print(f"  {condition}: {rate:.3f} [{lo:.3f}, {hi:.3f}]  ({n}/{total})")

    # Side-effect check: ablation should be SELECTIVE (fix quadrant C's
    # over-caution) not a general safety-breaker (quadrant A refusal
    # collapsing would mean the ablation is too blunt to be a clean H4 test).
    print("\n=== Side-effect check: quadrant A hard-refusal, baseline vs ablated ===")
    for condition in CONDITIONS:
        total = totals[condition]["A"]
        n = counts[condition]["A"]["refusal"]
        if total:
            ci_result = rate_with_ci(n, total)
            rate = ci_result["rate"]
            lo = ci_result["ci_low"]
            hi = ci_result["ci_high"]
            print(f"  {condition}: {rate:.3f} [{lo:.3f}, {hi:.3f}]  ({n}/{total})")

    summary = {
        condition: {
            quadrant: {
                "n": totals[condition][quadrant],
                **{cat: counts[condition][quadrant][cat] for cat in CATEGORIES},
            }
            for quadrant in QUADRANTS
        }
        for condition in CONDITIONS
    }
    out_path = Path("results/causal_ablation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved raw counts to {out_path}")


if __name__ == "__main__":
    main()