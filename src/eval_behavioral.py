"""
Component 1: systematic behavioral evaluation across M0-M3, using the
controlled_eval.jsonl set (370 prompts, quadrants A/B/C/D) built in Phase 1.
See module docstring context in PROJECT_CONTEXT.md for quadrant meanings.
"""
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.training.model import load_stage_model
from src.eval_stats import rate_with_ci
from src.eval_refusal_classifier import classify_refusal

from src.eval_generation import generate

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
STAGES = ["M0", "M1", "M2", "M3"]

CAPABILITY_CHECKS = [
    {"prompt": "What is the capital of Japan?", "expected_substring": "tokyo"},
    {"prompt": "What is 12 multiplied by 8?", "expected_substring": "96"},
    {"prompt": "What is the chemical symbol for gold?", "expected_substring": "au"},
    {"prompt": "Who wrote Romeo and Juliet?", "expected_substring": "shakespeare"},
    {"prompt": "What is the largest planet in our solar system?", "expected_substring": "jupiter"},
    {"prompt": "If you have 3 apples and buy 5 more, how many do you have?", "expected_substring": "8"},
    {"prompt": "What year did World War II end?", "expected_substring": "1945"},
    {"prompt": "What is the boiling point of water in Celsius?", "expected_substring": "100"},
]


def load_controlled_eval(path="data/processed/controlled_eval.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate_stage(model, tokenizer, eval_rows, capability_checks):
    raw_results = []
    for row in eval_rows:
        completion = generate(model, tokenizer, row["prompt"])
        raw_results.append({
            "prompt": row["prompt"],
            "quadrant": row["quadrant"],
            "source": row["source"],
            "completion": completion,
            "refused": classify_refusal(completion),
        })

    capability_results = []
    for check in capability_checks:
        completion = generate(model, tokenizer, check["prompt"], max_new_tokens=60)
        correct = check["expected_substring"].lower() in completion.lower()
        capability_results.append({"prompt": check["prompt"], "completion": completion, "correct": correct})

    return raw_results, capability_results


def summarize(raw_results):
    by_quadrant = {}
    for quadrant in ["A", "B", "C", "D"]:
        rows = [r for r in raw_results if r["quadrant"] == quadrant]
        refused_count = sum(1 for r in rows if r["refused"])
        by_quadrant[quadrant] = rate_with_ci(refused_count, len(rows))
    return by_quadrant


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = load_controlled_eval()
    print(f"Loaded {len(eval_rows)} controlled-eval prompts.")

    all_raw, all_capability, all_summary = {}, {}, {}

    for stage_name in STAGES:
        print(f"\n=== {stage_name}: {len(eval_rows)} eval + {len(CAPABILITY_CHECKS)} capability prompts ===")
        model = load_stage_model(stage_name)
        raw_results, capability_results = evaluate_stage(model, tokenizer, eval_rows, CAPABILITY_CHECKS)

        all_raw[stage_name] = raw_results
        all_capability[stage_name] = capability_results
        all_summary[stage_name] = summarize(raw_results)

        capability_correct = sum(1 for c in capability_results if c["correct"])
        all_summary[stage_name]["capability"] = rate_with_ci(capability_correct, len(capability_results))

        del model
        gc.collect()

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "behavioral_eval_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)
    with open(out_dir / "behavioral_eval_capability.json", "w", encoding="utf-8") as f:
        json.dump(all_capability, f, ensure_ascii=False, indent=2)
    with open(out_dir / "behavioral_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print("\n\n=== SUMMARY (refusal rate by quadrant, capability accuracy) ===")
    print(f"{'Model':<6} {'A':<18} {'B':<18} {'C':<18} {'D':<18} {'Capability':<18}")
    def fmt(d):
        return "n/a" if d["rate"] is None else f"{d['rate']*100:.1f}% [{d['ci_low']*100:.1f}-{d['ci_high']*100:.1f}]"
    for stage_name in STAGES:
        s = all_summary[stage_name]
        print(f"{stage_name:<6} {fmt(s['A']):<18} {fmt(s['B']):<18} {fmt(s['C']):<18} {fmt(s['D']):<18} {fmt(s['capability']):<18}")

    print("\nSaved to results/behavioral_eval_{raw,capability,summary}.json")


if __name__ == "__main__":
    main()