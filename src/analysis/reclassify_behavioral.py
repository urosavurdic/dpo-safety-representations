"""Re-apply the refusal classifier to already-generated responses.

No regeneration, no GPU. Reads the 654-row responses, falling back to the
pre-freeze file if they are absent, and writes labelled rows to a NEW file --
the 654-row responses are benchmark-bound and must never be rewritten in place.
"""
import json
from pathlib import Path

from src.common.refusal_classifier import classify_refusal, is_degenerate, is_soft_deflection
from src.common.stats import rate_with_ci

RESPONSES = Path("results/behavioral_eval/responses_654.json")
LEGACY_RESPONSES = Path("results/behavioral_eval/raw.json")
# Distinct output names. results/behavioral_eval/refusal_rates_370era.json is the
# committed PRE-FREEZE summary ("v2" there means the second classifier, not the
# 654-row era) and must not be overwritten by a 654-row run.
LABELLED_OUT = Path("results/behavioral_eval/reclassified_654.json")
SUMMARY_OUT = Path("results/behavioral_eval/refusal_rates_654.json")


def _response_text(row):
    """The 654-row files use "response"; pre-freeze files used "completion"."""
    text = row.get("response")
    return row.get("completion", "") if text is None else text


def load_responses():
    path = RESPONSES if RESPONSES.exists() else LEGACY_RESPONSES
    if not path.exists():
        raise FileNotFoundError(
            f"no responses found: looked for {RESPONSES} then {LEGACY_RESPONSES}"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + chr(10)
    path.write_text(payload, encoding="utf-8", newline="")


def main():
    source_path, all_raw = load_responses()

    all_summary = {}
    for stage_name, rows in all_raw.items():
        for row in rows:
            text = _response_text(row)
            row["refused"] = classify_refusal(text)
            row["degenerate"] = is_degenerate(text)
            row["soft_deflection"] = is_soft_deflection(text)

        by_quadrant = {}
        for quadrant in ["A", "B", "C", "D"]:
            quadrant_rows = [r for r in rows if r["quadrant"] == quadrant]
            refused_count = sum(1 for r in quadrant_rows if r["refused"])
            by_quadrant[quadrant] = rate_with_ci(refused_count, len(quadrant_rows))
        all_summary[stage_name] = by_quadrant

    _write(LABELLED_OUT, all_raw)
    _write(SUMMARY_OUT, all_summary)

    print(f"Read {source_path}; labelled rows -> {LABELLED_OUT}, summary -> {SUMMARY_OUT}")
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