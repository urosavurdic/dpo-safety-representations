"""
Samples completions for manual review against the classifier's predictions,
stratified across predicted-refuse/predicted-comply and across models.
"""
import json
import random
from pathlib import Path


def main(n_per_stratum=10, seed=42):
    with open("results/behavioral_eval_raw.json", "r", encoding="utf-8") as f:
        all_raw = json.load(f)

    rng = random.Random(seed)
    sample = []
    for stage_name, rows in all_raw.items():
        refused = [dict(r, stage=stage_name) for r in rows if r["refused"]]
        complied = [dict(r, stage=stage_name) for r in rows if not r["refused"]]
        sample += rng.sample(refused, min(n_per_stratum, len(refused)))
        sample += rng.sample(complied, min(n_per_stratum, len(complied)))

    rng.shuffle(sample)
    for row in sample:
        row["human_label"] = None  # fill in: true = you'd call this a refusal, false = not

    out_path = Path("results/classifier_validation_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(sample)} completions to {out_path}.")
    print("Fill in 'human_label' for each, then run: python -m src.check_classifier_agreement")


if __name__ == "__main__":
    main()