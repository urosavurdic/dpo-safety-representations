"""
Build M1's SFT-Helpful training data from Alpaca, explicitly excluding the
prompts reserved for quadrant D of the controlled eval set (see
src/build_eval_set.py). This is what keeps quadrant D genuinely held out.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.check_leakage import normalize

import argparse


def load_reserved_eval_prompts(path="data/processed/alpaca_reserved_for_eval.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_m1_dataset(rows, reserved_prompts, n_target=6000, seed=42):
    reserved_normalized = {normalize(p) for p in reserved_prompts}

    candidates = []
    for row in rows:
        if row["input"] != "":
            continue  # single-turn only, matches quadrant D filtering
        if normalize(row["instruction"]) in reserved_normalized:
            continue  # explicit exclusion of reserved eval prompts
        if not row["instruction"].strip() or not row["output"].strip():
            continue
        candidates.append({"prompt": row["instruction"].strip(), "response": row["output"].strip()})

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n_target]

    # Defensive check: true by construction, but assert it anyway - cheap
    # insurance against a future refactor silently breaking the exclusion.
    selected_normalized = {normalize(r["prompt"]) for r in selected}
    overlap = selected_normalized & reserved_normalized
    assert not overlap, f"Reserved eval prompts leaked into M1 training data: {overlap}"

    return selected


def save_jsonl(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def remove_flagged_prompts(data, flagged_prompts):
    flagged_normalized = {normalize(p) for p in flagged_prompts}
    return [r for r in data if normalize(r["prompt"]) not in flagged_normalized]


def load_flagged_training_prompts(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return [d["closest_train_prompt"] for d in report.get("near_duplicates", [])]


def load_exclusion_list(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def update_exclusion_list(path, new_prompts):
    """Merge new_prompts into the persistent exclusion list (union, deduped by
    normalized text), save, and return the merged list."""
    existing = load_exclusion_list(path)
    combined = {normalize(p): p for p in existing}
    for p in new_prompts:
        combined[normalize(p)] = p
    merged = list(combined.values())

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-report",
        default=None,
        help="Path to a dedup report; its flagged rows are merged into the persistent exclusion list.",
    )
    parser.add_argument(
        "--exclusion-list",
        default="data/processed/m1_near_dup_exclusions.json",
        help="Path to the persistent, accumulating near-duplicate exclusion list.",
    )
    args = parser.parse_args()

    print("Loading Alpaca (train split)...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"Loaded {len(dataset)} raw rows.")

    reserved = load_reserved_eval_prompts()
    print(f"Reserved quadrant-D eval prompts: {len(reserved)}")

    near_dup_exclusions = load_exclusion_list(args.exclusion_list)
    if args.exclude_report:
        newly_flagged = load_flagged_training_prompts(args.exclude_report)
        near_dup_exclusions = update_exclusion_list(args.exclusion_list, newly_flagged)
        print(f"Merged {len(newly_flagged)} newly flagged prompt(s) into the persistent "
              f"exclusion list (now {len(near_dup_exclusions)} total, ever found).")
    else:
        print(f"Using existing persistent exclusion list: {len(near_dup_exclusions)} entries.")

    all_exclusions = reserved + near_dup_exclusions
    m1_data = build_m1_dataset(dataset, all_exclusions, n_target=6000)
    print(f"Final M1 training set: {len(m1_data)} examples.")

    save_jsonl(m1_data, Path("data/processed/sft_helpful.jsonl"))
    print("Saved to data/processed/sft_helpful.jsonl")


if __name__ == "__main__":
    main()