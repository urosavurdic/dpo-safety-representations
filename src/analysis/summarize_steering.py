"""Summarize a steering raw-results file.

--file is required, and both the condition pairs and the output filename are
derived from the file's own contents rather than assumed. Steering files across
stages and configs share a row shape, so a wrong file produces a plausible
summary instead of an error.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.common.refusal_classifier import classify_completion
from src.common.stats import rate_with_ci
from src.common.io import load_json

CATEGORIES = ["degenerate", "refusal", "soft_deflection", "comply"]


def find_condition_pairs(rows):
    """Derive (baseline_stage, steered_stage) pairs from whatever stage names
    are actually present, instead of assuming a fixed "M3_baseline"/
    "M3_steered". A file may contain more than one pair (e.g. multiple
    layers/configs run into the same output path)."""
    stages = sorted({row["stage"] for row in rows})
    pairs = []
    for stage in stages:
        if stage.endswith("_baseline"):
            prefix = stage[: -len("_baseline")]
            steered = f"{prefix}_steered"
            if steered in stages:
                pairs.append((stage, steered))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True,
                         help="Raw steering results file, e.g. results/raw/steering_654_M3_L24_...json. "
                              "No default on purpose - always say explicitly which file you mean.")
    parser.add_argument("--out", default=None,
                         help="Output path. Defaults to results/summaries/<input filename>_summary.json "
                              "(derived from --file, never a fixed name).")
    args = parser.parse_args()
    rows = load_json(args.file)
    print(f"Loaded {len(rows)} rows from {args.file}.")

    pairs = find_condition_pairs(rows)
    if not pairs:
        print("No baseline/steered condition pairs found (no stage name ends in "
              "'_baseline' with a matching '_steered' counterpart). Nothing to summarize.")
        return

    quadrants = sorted({row["quadrant"] for row in rows})
    summary = {}
    for baseline_stage, steered_stage in pairs:
        print(f"\n=== {baseline_stage} vs {steered_stage} ===")
        for q in quadrants:
            counts = defaultdict(lambda: defaultdict(int))
            totals = defaultdict(int)
            for row in rows:
                if row["stage"] not in (baseline_stage, steered_stage) or row["quadrant"] != q:
                    continue
                counts[row["stage"]][classify_completion(row["response"])] += 1
                totals[row["stage"]] += 1

            if totals[baseline_stage] == 0 and totals[steered_stage] == 0:
                continue
            print(f"  quadrant {q}:")
            for stage in (baseline_stage, steered_stage):
                total = totals[stage]
                if total == 0:
                    print(f"    {stage}: NO ROWS")
                    continue
                print(f"    {stage} (n={total}):")
                for category in CATEGORIES:
                    n = counts[stage][category]
                    ci = rate_with_ci(n, total)
                    print(f"      {category:16s} {ci['rate']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  ({n}/{total})")
                summary[f"{stage}_{q}"] = {"n": total, **{cat: counts[stage][cat] for cat in CATEGORIES}}

    out_path = Path(args.out) if args.out else Path("results/summaries") / f"{Path(args.file).stem}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
