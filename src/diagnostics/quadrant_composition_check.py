"""
Checks whether the controlled eval set's own design assumptions actually
hold in the real, built data - not just asserted, measured. Motivated
directly by a project review question: is there a way to check the eval
set is doing what it's meant to, or are we just trusting the design?

Three predictions the quadrant design implies, that this can actually test:
  1. B (benign, harmful-SOUNDING wording) should score comparably to A on
     surface cue density, DESPITE being benign - that's the entire point of
     B (XSTest's "sounds risky, isn't" trap). If B scores near D instead,
     the eval set isn't actually testing what it claims to.
  2. C (harmful, reduced-cue) should score much lower than A on cue
     density - the whole point of the rewording effort. If it doesn't,
     "reduced-cue" isn't actually true of the built data, regardless of
     what the pipeline's own pre-screen said about individual candidates.
  3. C's cue density should approach D's (both near the "neutral wording"
     floor) while still being ground-truth harmful - if C sits far above
     D, the wording reduction hasn't gone far enough; if C matches D
     closely, that's evidence the reduction genuinely worked at the
     dataset level, not just candidate-by-candidate.

This measures ONE narrow thing (generic wrongdoing/evasion cue-word
density, via score_lexical_risk_cues's predeclared lexicon) plus basic
length statistics - not a full semantic-neutrality audit. A quadrant
passing this check is evidence the design assumption holds on this
narrow axis, not proof the eval set is doing everything it's meant to.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, median

from src.diagnostics.score_lexical_risk_cues import score_prompt


def load_eval_set(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_quadrant_stats(rows):
    """rows: list of {"prompt": ..., "quadrant": ...}. Returns
    {quadrant: {n, mean_words, median_words, mean_cue_hits, pct_with_cue_hit}}."""
    by_quadrant = {}
    for row in rows:
        by_quadrant.setdefault(row["quadrant"], []).append(row["prompt"])

    stats = {}
    for quadrant, prompts in by_quadrant.items():
        word_counts = [len(p.split()) for p in prompts]
        cue_hits = [score_prompt(p)[0] for p in prompts]
        stats[quadrant] = {
            "n": len(prompts),
            "mean_words": round(mean(word_counts), 1),
            "median_words": median(word_counts),
            "mean_cue_hits": round(mean(cue_hits), 3),
            "pct_with_cue_hit": round(100 * sum(1 for h in cue_hits if h > 0) / len(cue_hits), 1),
        }
    return stats


def check_design_predictions(stats):
    """Returns a list of (prediction, holds: bool, detail) - explicit,
    checkable claims, not vibes. Doesn't fail loudly on violation (this is
    a diagnostic, not a test suite) - reports honestly either way."""
    results = []
    if "A" in stats and "B" in stats:
        a, b = stats["A"]["mean_cue_hits"], stats["B"]["mean_cue_hits"]
        holds = b >= a * 0.5  # B should be comparably high, not near zero - loose threshold, this is directional
        results.append((
            "B (benign, harmful-sounding) scores comparably to A on cue density despite being benign",
            holds, f"A={a}, B={b}",
        ))
    if "A" in stats and "C" in stats:
        a, c = stats["A"]["mean_cue_hits"], stats["C"]["mean_cue_hits"]
        holds = c < a
        results.append((
            "C (harmful, reduced-cue) scores lower than A on cue density",
            holds, f"A={a}, C={c}",
        ))
    if "C" in stats and "D" in stats:
        c, d = stats["C"]["mean_cue_hits"], stats["D"]["mean_cue_hits"]
        gap = round(c - d, 3)
        results.append((
            "C's cue density approaches D's (both near the neutral-wording floor)",
            gap < 0.3,  # loose, directional threshold - the point is "close", not exact equality
            f"C={c}, D={d}, gap={gap}",
        ))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/processed/controlled_eval.jsonl")
    parser.add_argument("--report", default="data/quadrant_composition_report.json")
    args = parser.parse_args()

    rows = load_eval_set(args.eval_set)
    stats = compute_quadrant_stats(rows)
    predictions = check_design_predictions(stats)

    print("Per-quadrant stats:")
    for q in sorted(stats):
        s = stats[q]
        print(f"  {q}: n={s['n']}, mean_words={s['mean_words']}, "
              f"mean_cue_hits={s['mean_cue_hits']}, pct_with_cue_hit={s['pct_with_cue_hit']}%")

    print("\nDesign predictions:")
    for prediction, holds, detail in predictions:
        status = "HOLDS" if holds else "DOES NOT HOLD"
        print(f"  [{status}] {prediction} ({detail})")

    report = {"stats": stats, "predictions": [
        {"prediction": p, "holds": h, "detail": d} for p, h, d in predictions
    ]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {args.report}")


if __name__ == "__main__":
    main()
