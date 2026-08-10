"""
Re-applies the (fixed) refusal classifier to already-generated completions,
without regenerating anything. Cheap, local, no GPU needed.
"""
import json
from pathlib import Path

from src.eval_refusal_classifier import classify_refusal
from src.eval_stats import rate_with_ci


def main():
    with open("results/behavioral_eval_raw.json", "r", encoding="utf-8") as f:
        all_raw = json.load(f)

    all_summary = {}
    for stage_name, rows in all_raw.items():
        for row in rows:
            row["refused"] = classify_refusal(row["completion"])

        by_quadrant = {}
        for quadrant in ["A", "B", "C", "D"]:
            quadrant_rows = [r for r in rows if r["quadrant"] == quadrant]
            refused_count = sum(1 for r in quadrant_rows if r["refused"])
            by_quadrant[quadrant] = rate_with_ci(refused_count, len(quadrant_rows))
        all_summary[stage_name] = by_quadrant

    with open("results/behavioral_eval_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)
    with open("results/behavioral_eval_summary_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print("Reclassified in place. New summary saved to results/behavioral_eval_summary_v2.json")
    print(f"{'Model':<6} {'A':<18} {'B':<18} {'C':<18} {'D':<18}")
    def fmt(d):
        return "n/a" if d["rate"] is None else f"{d['rate']*100:.1f}% [{d['ci_low']*100:.1f}-{d['ci_high']*100:.1f}]"
    for stage_name, s in all_summary.items():
        print(f"{stage_name:<6} {fmt(s['A']):<18} {fmt(s['B']):<18} {fmt(s['C']):<18} {fmt(s['D']):<18}")


if __name__ == "__main__":
    main()