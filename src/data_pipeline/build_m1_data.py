"""
Build M1 (or M1_alt)'s SFT-Helpful training data, explicitly excluding the
prompts reserved for quadrant D of the controlled eval set (see
src/build_eval_set.py). This is what keeps quadrant D genuinely held out -
true regardless of which source dataset M1/M1_alt is built from, since
quadrant D's prompts are the reserved set, not something specific to Alpaca.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.diagnostics.check_leakage import normalize

import argparse

# Per-source loaders, normalized to a common {"instruction", "input", "output"}
# shape so build_m1_dataset() (selection/exclusion/dedup logic) never needs to
# know which source it's looking at. Add a new entry here for any future
# alternative M1 dataset rather than writing a second build_m1_dataset().
SOURCES = {
    "alpaca": {
        "hf_id": "tatsu-lab/alpaca",
        "split": "train",
        # Alpaca rows are already in the target shape.
        "normalize_row": lambda row: {
            "instruction": row["instruction"], "input": row["input"], "output": row["output"],
        },
    },
    "dolly": {
        "hf_id": "databricks/databricks-dolly-15k",
        "split": "train",
        # Dolly's columns are instruction/context/response/category - context
        # plays the same "extra input beyond the instruction" role Alpaca's
        # `input` does, so single-turn filtering (row["input"] != "") still
        # applies unchanged once mapped through this normalizer.
        "normalize_row": lambda row: {
            "instruction": row["instruction"], "input": row["context"], "output": row["response"],
        },
    },
}


def load_source_rows(source: str):
    spec = SOURCES[source]
    dataset = load_dataset(spec["hf_id"], split=spec["split"])
    return [spec["normalize_row"](row) for row in dataset]


def load_reserved_eval_prompts(path="data/processed/quadrant_d_reserved_for_eval.json"):
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
        "--dataset", default="alpaca", choices=list(SOURCES.keys()),
        help="Source dataset for M1's SFT-helpful data (default: alpaca, unchanged behavior).",
    )
    parser.add_argument(
        "--exclude-report",
        default=None,
        help="Path to a dedup report; its flagged rows are merged into the persistent exclusion list.",
    )
    parser.add_argument(
        "--exclusion-list",
        default=None,
        help="Path to the persistent, accumulating near-duplicate exclusion list. "
             "Defaults to data/processed/m1_near_dup_exclusions.json for alpaca, "
             "data/processed/m1_alt_near_dup_exclusions.json for other sources - "
             "kept separate per source so one source's flagged near-dups never "
             "silently exclude prompts from a different source's candidate pool.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output jsonl path. Defaults to data/processed/sft_helpful.jsonl for alpaca "
             "(unchanged), data/processed/sft_helpful_alt.jsonl for other sources.",
    )
    args = parser.parse_args()

    exclusion_list_path = args.exclusion_list or (
        "data/processed/m1_near_dup_exclusions.json" if args.dataset == "alpaca"
        else "data/processed/m1_alt_near_dup_exclusions.json"
    )
    output_path = args.output or (
        "data/processed/sft_helpful.jsonl" if args.dataset == "alpaca"
        else "data/processed/sft_helpful_alt.jsonl"
    )

    print(f"Loading {args.dataset} (train split)...")
    rows = load_source_rows(args.dataset)
    print(f"Loaded {len(rows)} raw rows.")

    reserved = load_reserved_eval_prompts()
    print(f"Reserved quadrant-D eval prompts: {len(reserved)}")

    near_dup_exclusions = load_exclusion_list(exclusion_list_path)
    if args.exclude_report:
        newly_flagged = load_flagged_training_prompts(args.exclude_report)
        near_dup_exclusions = update_exclusion_list(exclusion_list_path, newly_flagged)
        print(f"Merged {len(newly_flagged)} newly flagged prompt(s) into the persistent "
              f"exclusion list (now {len(near_dup_exclusions)} total, ever found).")
    else:
        print(f"Using existing persistent exclusion list ({exclusion_list_path}): "
              f"{len(near_dup_exclusions)} entries.")

    all_exclusions = reserved + near_dup_exclusions
    m1_data = build_m1_dataset(rows, all_exclusions, n_target=6000)
    print(f"Final M1 ({args.dataset}) training set: {len(m1_data)} examples.")

    save_jsonl(m1_data, Path(output_path))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()