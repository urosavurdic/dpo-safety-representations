"""
Diagnostic: harm-category coverage of our M2/M3 training data. No GPU
needed - re-reads PKU-SafeRLHF (cached locally) and cross-references by
exact prompt text against dpo_pairs.jsonl, to test whether categories like
vehicle-related harm are underrepresented - a candidate explanation for why
M2 and M3 both comply with a hotwiring request while correctly refusing a
password-stealing-virus request.
"""
import json
from collections import Counter

from datasets import load_dataset


def load_our_prompts(path="data/processed/dpo_pairs.jsonl"):
    prompts = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.add(row["prompt"].strip())
    return prompts


def main():
    our_prompts = load_our_prompts()
    print(f"Our M2/M3 training set: {len(our_prompts)} unique prompts.")

    print("Loading PKU-SafeRLHF (train split) to recover harm categories...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    full_pool_categories = Counter()
    our_sample_categories = Counter()

    for row in dataset:
        if row["is_response_0_safe"] == row["is_response_1_safe"]:
            continue  # matches our original safety-disagreement filter

        safer_id = row["safer_response_id"]
        unsafe_id = 1 - safer_id  # harm category lives on the unsafe response
        categories = row[f"response_{unsafe_id}_harm_category"]
        active_categories = [cat for cat, flagged in categories.items() if flagged]

        for cat in active_categories:
            full_pool_categories[cat] += 1

        if row["prompt"].strip() in our_prompts:
            for cat in active_categories:
                our_sample_categories[cat] += 1

    print("\nCategory coverage: our sample vs. full eligible pool")
    print(f"{'Category':<40} {'Our sample':>12} {'Full pool':>12} {'Our %':>8}")
    for cat, full_count in full_pool_categories.most_common():
        our_count = our_sample_categories.get(cat, 0)
        pct = 100 * our_count / full_count if full_count else 0
        print(f"{cat:<40} {our_count:>12} {full_count:>12} {pct:>7.1f}%")


if __name__ == "__main__":
    main()