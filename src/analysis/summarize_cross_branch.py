"""
Cross-branch comparison: does each original stage's finding replicate on
the alt (Dolly-initialized M1) branch? Reads only already-computed results
- no GPU, no new generation/extraction/probing. Answers the question the
whole alt branch exists to answer: is a given finding dataset-specific or
not.

Reuses classify_refusal/is_soft_deflection/is_degenerate directly against
results/behavioral_eval/raw.json (not summary_v2.json, which only persists
the plain refusal rate - soft-deflection is the actual headline metric
throughout this project and was previously only ever printed to stdout by
reclassify_behavioral.py, never saved). Probe and direction comparisons
read straight from their own already-saved JSON.
"""
import json
from pathlib import Path

from src.analysis.eval_refusal_classifier import classify_refusal, is_degenerate, is_soft_deflection
from src.eval_stats import rate_with_ci

CROSS_BRANCH_PAIRS = [
    ("M1", "M1_alt"),
    ("M2", "M2_alt"),
    ("M3", "M3_alt"),
    ("M3_direct", "M3_direct_alt"),
]


def load_raw_behavioral(path="results/behavioral_eval/raw.json"):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def behavioral_rates_for_stage(raw_rows_by_stage, stage):
    """Per-quadrant refusal/soft_deflection/degenerate rates with Wilson
    CIs, or None if this stage has no behavioral results yet."""
    if stage not in raw_rows_by_stage:
        return None
    rows = raw_rows_by_stage[stage]
    out = {}
    for quadrant in ["A", "B", "C", "D"]:
        q_rows = [r for r in rows if r["quadrant"] == quadrant]
        if not q_rows:
            continue
        refused = sum(1 for r in q_rows if classify_refusal(r["completion"]))
        soft_defl = sum(1 for r in q_rows if is_soft_deflection(r["completion"]))
        degenerate = sum(1 for r in q_rows if is_degenerate(r["completion"]))
        out[quadrant] = {
            "refusal": rate_with_ci(refused, len(q_rows)),
            "soft_deflection": rate_with_ci(soft_defl, len(q_rows)),
            "degenerate": rate_with_ci(degenerate, len(q_rows)),
        }
    return out


def probe_best_layer_for_stage(stage, probes_dir="results/probes"):
    path = Path(probes_dir) / f"{stage}_probe_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    return max(results, key=lambda r: r["cv_accuracy_mean"])


def direction_cross_branch_similarity(orig, alt, cosine_path="results/refusal_direction/cosine_similarity.json"):
    path = Path(cosine_path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    values = data.get("cross_branch", {}).get(f"{orig}_vs_{alt}")
    if values is None:
        return None
    return {"per_layer": values, "mean": sum(values) / len(values)}


def build_comparison(orig, alt, raw_rows_by_stage):
    """Returns a dict of whatever's available for this pair - never raises
    on missing data, so this can be run at any point in a staggered,
    multi-session alt-branch training schedule."""
    result = {"pair": f"{orig}_vs_{alt}"}

    orig_beh = behavioral_rates_for_stage(raw_rows_by_stage, orig)
    alt_beh = behavioral_rates_for_stage(raw_rows_by_stage, alt)
    if orig_beh and alt_beh:
        result["behavioral"] = {"orig": orig_beh, "alt": alt_beh}

    orig_probe = probe_best_layer_for_stage(orig)
    alt_probe = probe_best_layer_for_stage(alt)
    if orig_probe and alt_probe:
        result["probes"] = {"orig": orig_probe, "alt": alt_probe}

    direction = direction_cross_branch_similarity(orig, alt)
    if direction:
        result["direction"] = direction

    return result


def print_comparison(comp):
    orig, alt = comp["pair"].split("_vs_")
    print(f"=== {orig} vs {alt} ===")

    if "behavioral" in comp:
        for quadrant, metric in [("A", "refusal"), ("C", "soft_deflection")]:
            o = comp["behavioral"]["orig"].get(quadrant, {}).get(metric)
            a = comp["behavioral"]["alt"].get(quadrant, {}).get(metric)
            if o and a and o["rate"] is not None and a["rate"] is not None:
                print(f"  Behavioral (quadrant {quadrant}, {metric}): "
                      f"{orig} {o['rate']:.1%} [{o['ci_low']:.1%},{o['ci_high']:.1%}] vs "
                      f"{alt} {a['rate']:.1%} [{a['ci_low']:.1%},{a['ci_high']:.1%}]  "
                      f"(delta {a['rate']-o['rate']:+.1%})")
    else:
        print("  Behavioral: not yet available for both sides")

    if "probes" in comp:
        op, ap = comp["probes"]["orig"], comp["probes"]["alt"]
        print(f"  Probes (best-layer CV acc): {orig} {op['cv_accuracy_mean']:.3f} vs {alt} {ap['cv_accuracy_mean']:.3f}")
        print(f"  Probes (quadrant C flagged unsafe): {orig} {op['quadrant_c_flagged_unsafe_frac']:.3f} vs "
              f"{alt} {ap['quadrant_c_flagged_unsafe_frac']:.3f}")
    else:
        print("  Probes: not yet available for both sides")

    if "direction" in comp:
        print(f"  Direction cosine similarity (mean across layers): {comp['direction']['mean']:.3f}")
    else:
        print("  Direction: not yet available")
    print()


def main():
    raw_rows_by_stage = load_raw_behavioral()
    print("Cross-branch comparison: original vs. alt (Dolly-initialized M1) branch\n")

    all_comparisons = {}
    for orig, alt in CROSS_BRANCH_PAIRS:
        comp = build_comparison(orig, alt, raw_rows_by_stage)
        print_comparison(comp)
        all_comparisons[comp["pair"]] = comp

    out_path = Path("results/summaries/cross_branch_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
