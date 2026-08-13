import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.analysis.summarize_causal_ablation import classify_completion
from src.eval_stats import rate_with_ci
from src.io_utils import load_json

CONDITIONS = ["M3_baseline", "M3_steered"]
CATEGORIES = ["degenerate", "refusal", "soft_deflection", "comply"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results/raw/steering_raw_D.json")
    args = parser.parse_args()
    rows = load_json(args.file)
    print(f"Loaded {len(rows)} rows.")

    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for row in rows:
        stage = row["stage"]
        counts[stage][classify_completion(row["response"])] += 1
        totals[stage] += 1

    print("\n=== Quadrant D (benign), baseline vs steered ===")
    for condition in CONDITIONS:
        total = totals[condition]
        if total == 0:
            print(f"  {condition}: NO ROWS")
            continue
        print(f"  {condition} (n={total}):")
        for category in CATEGORIES:
            n = counts[condition][category]
            ci = rate_with_ci(n, total)
            print(f"    {category:16s} {ci['rate']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  ({n}/{total})")

    summary = {c: {"n": totals[c], **{cat: counts[c][cat] for cat in CATEGORIES}} for c in CONDITIONS}
    out_path = Path("results/summaries/steering_D_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()