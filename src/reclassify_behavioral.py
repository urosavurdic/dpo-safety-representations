"""
Re-applies the (fixed) refusal classifier to already-generated completions,
without regenerating anything. Cheap, local, no GPU needed.
"""
import json
from pathlib import Path

from src.eval_refusal_classifier import classify_refusal, is_degenerate, is_soft_deflection
from src.eval_stats import rate_with_ci


def main():
    with open("results/behavioral_eval_raw.json", "r", encoding="utf-8") as f:
        all_raw = json.load(f)

    all_summary = {}
    for stage_name, rows in all_raw.items():
        for row in rows:
            row["refused"] = classify_refusal(row["completion"])
            row["degenerate"] = is_degenerate(row["completion"])
            row["soft_deflection"] = is_soft_deflection(row["completion"])

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

    print("\nDegenerate-output rate by model (garbage/repetition, not a real refusal or answer):")
    for stage_name, rows in all_raw.items():
        degenerate_count = sum(1 for r in rows if r["degenerate"])
        d = rate_with_ci(degenerate_count, len(rows))
        print(f"  {stage_name}: {d['rate']*100:.1f}% [{d['ci_low']*100:.1f}-{d['ci_high']*100:.1f}]")

    print("\nRefusal rate among non-degenerate completions only (fairer comparison for M0):")
    print(f"{'Model':<6} {'A':<18} {'B':<18} {'C':<18} {'D':<18}")
    for stage_name, rows in all_raw.items():
        coherent = [r for r in rows if not r["degenerate"]]
        row_fmt = []
        for quadrant in ["A", "B", "C", "D"]:
            quadrant_rows = [r for r in coherent if r["quadrant"] == quadrant]
            refused_count = sum(1 for r in quadrant_rows if r["refused"])
            d = rate_with_ci(refused_count, len(quadrant_rows))
            row_fmt.append("n/a" if d["rate"] is None else f"{d['rate']*100:.1f}% [{d['ci_low']*100:.1f}-{d['ci_high']*100:.1f}]")
        print(f"{stage_name:<6} {row_fmt[0]:<18} {row_fmt[1]:<18} {row_fmt[2]:<18} {row_fmt[3]:<18}")

    print("\nSoft-deflection rate by model (downplays feasibility, no real refusal phrase, no real operational content):")
    for stage_name, rows in all_raw.items():
        c_rows = [r for r in rows if r["quadrant"] == "C"]
        deflect_count = sum(1 for r in c_rows if r["soft_deflection"])
        d = rate_with_ci(deflect_count, len(c_rows))
        print(f"  {stage_name} (quadrant C only): {d['rate']*100:.1f}% [{d['ci_low']*100:.1f}-{d['ci_high']*100:.1f}]")

    print("\nSoft-deflection rate by model, all quadrants (B/D should stay near zero - this is the over-refusal check):")
    print(f"{'Model':<6} {'A':<18} {'B':<18} {'C':<18} {'D':<18}")
    for stage_name, rows in all_raw.items():
        row_fmt = []
        for quadrant in ["A", "B", "C", "D"]:
            quadrant_rows = [r for r in rows if r["quadrant"] == quadrant]
            deflect_count = sum(1 for r in quadrant_rows if r["soft_deflection"])
            d = rate_with_ci(deflect_count, len(quadrant_rows))
            row_fmt.append("n/a" if d["rate"] is None else f"{d['rate']*100:.1f}% [{d['ci_low']*100:.1f}-{d['ci_high']*100:.1f}]")
        print(f"{stage_name:<6} {row_fmt[0]:<18} {row_fmt[1]:<18} {row_fmt[2]:<18} {row_fmt[3]:<18}")

        
if __name__ == "__main__":
    main()