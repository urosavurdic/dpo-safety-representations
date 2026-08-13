"""
Build matched DPO and SFT-safety datasets from PKU-SafeRLHF.

Design decision (see PROJECT_CONTEXT.md): M2 (SFT-safety) and M3 (DPO) must be
trained on the SAME prompts with the SAME "correct" response, so that any
representational difference between M2 and M3 is caused by the training
OBJECTIVE, not by different training DATA. We enforce that here by deriving
both datasets from one filtered pass over PKU-SafeRLHF.

We additionally filter to rows where is_response_0_safe != is_response_1_safe,
so every pair is a genuine safe-vs-unsafe contrast, not a same-safety-class
preference judgment.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset


def load_pku_safe_rlhf(split: str = "train"):
    """Load PKU-SafeRLHF from Hugging Face. Requires internet access."""
    return load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)


def build_matched_pairs(rows, n_target: int = 4000, seed: int = 42):
    """
    rows: iterable of dicts with at least the PKU-SafeRLHF schema fields.
    Returns (dpo_pairs, sft_examples), both lists of dicts, same prompts,
    length <= n_target.
    """
    candidates = []
    for row in rows:
        # Only keep rows where the two responses disagree on safety —
        # this is the "clean safety contrast" filter described above.
        if row["is_response_0_safe"] == row["is_response_1_safe"]:
            continue

        safer_id = row["safer_response_id"]
        chosen = row[f"response_{safer_id}"]
        rejected = row[f"response_{1 - safer_id}"]

        # Skip degenerate rows (empty strings do occur in this dataset per
        # its own schema notes — response fields can have length 0).
        if not chosen.strip() or not rejected.strip() or not row["prompt"].strip():
            continue

        candidates.append(
            {
                "prompt": row["prompt"].strip(),
                "chosen": chosen.strip(),
                "rejected": rejected.strip(),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n_target]

    dpo_pairs = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in selected
    ]
    sft_examples = [
        {"prompt": r["prompt"], "response": r["chosen"]} for r in selected
    ]

    return dpo_pairs, sft_examples


def save_jsonl(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print("Loading PKU-SafeRLHF (train split)...")
    dataset = load_pku_safe_rlhf("train")
    print(f"Loaded {len(dataset)} raw rows.")

    dpo_pairs, sft_examples = build_matched_pairs(dataset, n_target=4000)

    print(f"After safety-disagreement filter + subsample: {len(dpo_pairs)} matched pairs.")
    if len(dpo_pairs) < 4000:
        print(
            f"NOTE: fewer than 4000 rows passed the filter "
            f"({len(dpo_pairs)} available) — using all of them. "
            f"This is fine, just noting it for the record."
        )

    out_dir = Path("data/processed")
    save_jsonl(dpo_pairs, out_dir / "dpo_pairs.jsonl")
    save_jsonl(sft_examples, out_dir / "sft_safety.jsonl")
    print(f"Saved to {out_dir}/dpo_pairs.jsonl and {out_dir}/sft_safety.jsonl")


if __name__ == "__main__":
    main()