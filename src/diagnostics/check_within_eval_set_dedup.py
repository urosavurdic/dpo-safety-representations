"""
Check for duplicate/near-duplicate prompts WITHIN the controlled eval set
itself - not eval-vs-training (that's check_leakage.py), but eval-vs-eval.

Motivated by a real, concrete pattern: check_leakage.py's Alpaca-training
report flagged "What is the legal drinking age in the USA?" as a near-dupe
of an Alpaca training prompt, and the SAME question (worded slightly
differently) also got flagged in the Dolly-training report. That's a sign
the SAME underlying question may have been drawn independently into more
than one of quadrant D's three sub-sources (Alpaca / Dolly / independent) -
which would mean two of the "150 items, three different sources" aren't
actually testing three different things, just the same generic trivia
question phrased three ways. Quietly reduces quadrant D's real diversity
without showing up as an error anywhere.

Reuses check_leakage.py's exact normalize/find_exact_duplicates/
find_near_duplicates functions rather than reimplementing - this is the
same kind of check, just pointed at two eval-set slices instead of an
eval-set slice and a training file.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.diagnostics.check_leakage import find_exact_duplicates, find_near_duplicates


def load_eval_set(path="data/processed/controlled_eval.jsonl"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group_by_source(rows, quadrant):
    """Splits a quadrant's rows by their "source" field (e.g. quadrant D's
    "Alpaca" / "Dolly-15k" / "OASST1"). Any quadrant with >1 distinct
    source is worth checking pairwise - currently that's just D, but this
    doesn't hardcode "D" so it stays useful if that ever changes."""
    by_source = defaultdict(list)
    for r in rows:
        if r["quadrant"] == quadrant:
            by_source[r["source"]].append(r["prompt"])
    return dict(by_source)


def check_all_pairs_within_quadrant(rows, quadrant, threshold=0.9):
    by_source = group_by_source(rows, quadrant)
    sources = sorted(by_source.keys())
    if len(sources) < 2:
        print(f"Quadrant {quadrant}: only {len(sources)} source(s) ({sources}) - nothing to compare.")
        return {}

    results = {}
    for i, source_a in enumerate(sources):
        for source_b in sources[i + 1:]:
            prompts_a, prompts_b = by_source[source_a], by_source[source_b]
            exact = find_exact_duplicates(prompts_a, prompts_b)
            near = find_near_duplicates(prompts_a, prompts_b, threshold=threshold)
            key = f"{source_a}_vs_{source_b}"
            results[key] = {
                "n_a": len(prompts_a), "n_b": len(prompts_b),
                "exact_duplicates": exact, "near_duplicates": near,
            }
            print(f"\n{quadrant}: {source_a} ({len(prompts_a)}) vs {source_b} ({len(prompts_b)})")
            print(f"  exact duplicates: {len(exact)}")
            print(f"  near duplicates (cosine >= {threshold}): {len(near)}")
            for d in near:
                print(f"    {d['eval_prompt']!r} ~= {d['closest_train_prompt']!r} (sim={d['similarity']})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/processed/controlled_eval.jsonl")
    parser.add_argument("--quadrant", default="D", help="Which quadrant's sources to check pairwise (default D - "
                                                          "the only quadrant with multiple sources right now)")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--report", default="data/dedup_report_within_quadrant_d.json")
    args = parser.parse_args()

    rows = load_eval_set(args.eval_set)
    results = check_all_pairs_within_quadrant(rows, args.quadrant, args.threshold)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {args.report}")

    any_flagged = any(r["exact_duplicates"] or r["near_duplicates"] for r in results.values())
    if any_flagged:
        print("WARNING: within-quadrant near/exact duplicates found - review before treating "
              "quadrant D's three sub-sources as independent.")
    else:
        print("No within-quadrant duplicates found across quadrant D's sub-sources.")


if __name__ == "__main__":
    main()
