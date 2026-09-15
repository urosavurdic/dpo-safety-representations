"""
Pulls all quadrant C completions for M1-M3 directly for manual review.
Quadrant C hitting 0% refusal for all three trained models simultaneously
is either a real finding (lexical-shortcut behavior - the core research
question) or a classifier miss - with only 20 examples per model, cheaper
to just read all of them than guess.
"""
import json


def main():
    with open("results/behavioral_eval/raw.json", "r", encoding="utf-8") as f:
        all_raw = json.load(f)

    for stage_name in ["M1", "M2", "M3"]:
        print(f"\n{'='*20} {stage_name} - Quadrant C {'='*20}")
        rows = [r for r in all_raw[stage_name] if r["quadrant"] == "C"]
        for r in rows:
            print(f"\nPrompt: {r['prompt']}")
            print(f"Refused={r['refused']} Degenerate={r['degenerate']}")
            print(f"Completion: {r['completion'][:300]}")


if __name__ == "__main__":
    main()