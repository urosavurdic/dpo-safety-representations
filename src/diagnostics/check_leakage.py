"""
Check for exact and near-duplicate overlap between the controlled eval set
and training data. The eval set must be genuinely held out - this script is
how we verify that claim rather than assume it.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import argparse


def load_prompts(jsonl_path, field="prompt"):
    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row[field])
    return prompts


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def find_exact_duplicates(eval_prompts, train_prompts):
    """Eval prompts that exactly match (after normalization) a training prompt."""
    train_normalized = {normalize(p) for p in train_prompts}
    return [p for p in eval_prompts if normalize(p) in train_normalized]


def flag_near_duplicates_from_similarity(eval_prompts, train_prompts, sim_matrix, threshold=0.9):
    """
    sim_matrix: shape (len(eval_prompts), len(train_prompts)), cosine similarities.
    Returns ALL (eval, train) pairs whose similarity exceeds threshold — not just
    the single best match per eval prompt. An eval prompt can appear more than
    once if it has multiple close training matches.
    """
    flagged = []
    for i, eval_prompt in enumerate(eval_prompts):
        for j, train_prompt in enumerate(train_prompts):
            sim = float(sim_matrix[i][j])
            if sim >= threshold:
                flagged.append(
                    {
                        "eval_prompt": eval_prompt,
                        "closest_train_prompt": train_prompt,
                        "similarity": round(sim, 4),
                    }
                )
    return flagged


def find_near_duplicates(eval_prompts, train_prompts, threshold=0.9, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    eval_emb = model.encode(eval_prompts, normalize_embeddings=True, show_progress_bar=False)
    train_emb = model.encode(train_prompts, normalize_embeddings=True, show_progress_bar=False)
    sim_matrix = eval_emb @ train_emb.T
    return flag_near_duplicates_from_similarity(eval_prompts, train_prompts, sim_matrix, threshold)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/dpo_pairs.jsonl")
    parser.add_argument("--report", default="data/dedup_report.json")
    parser.add_argument("--train-label", default="PKU-SafeRLHF derived")
    args = parser.parse_args()

    eval_prompts = load_prompts("data/processed/controlled_eval.jsonl")
    train_prompts = load_prompts(args.train)

    print(f"Eval prompts: {len(eval_prompts)}")
    print(f"Training prompts ({args.train_label}): {len(train_prompts)}")

    exact_dupes = find_exact_duplicates(eval_prompts, train_prompts)
    print(f"Exact duplicates: {len(exact_dupes)}")

    print("Computing embeddings for near-duplicate check...")
    near_dupes = find_near_duplicates(eval_prompts, train_prompts, threshold=0.9)
    print(f"Near-duplicates (cosine >= 0.9): {len(near_dupes)}")

    report = {
        "n_eval_prompts": len(eval_prompts),
        "n_train_prompts": len(train_prompts),
        "train_source": args.train_label,
        "exact_duplicates": exact_dupes,
        "near_duplicates": near_dupes,
    }

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to {out_path}")
    if exact_dupes or near_dupes:
        print("WARNING: potential leakage detected — review the report before training.")
    else:
        print(f"No leakage detected between eval set and {args.train_label} training data.")


if __name__ == "__main__":
    main()