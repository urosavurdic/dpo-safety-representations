"""
Component 3, corrected framing (PROJECT_CONTEXT.md decision #22).

CV accuracy on the A-vs-B(subset) training contrast saturates near 1.0 at
nearly every layer, for every stage - including untrained M0 (confirmed via
src/diagnose_probe_layers.py). A linear probe latches onto dataset/style
fingerprint almost immediately and stays near ceiling regardless of network
depth - not a usable metric, retired as Component 3's headline result.

What carries real, stage-dependent signal: how much of each held-out
quadrant the trained A-vs-B decision boundary flags "unsafe," at the FINAL
layer (28) - the representation immediately before generation. Reads the
already-computed results/probes/{stage}_probe_results.json files from
eval_probes.py - no retraining, no GPU.
"""
import json

from src.eval_stats import rate_with_ci

STAGES = ["M0", "M1", "M2", "M3", "M3_direct"]
FINAL_LAYER = 28
QUADRANTS = {
    "holdout_b_flagged_unsafe_frac": ("B (held-out)", 200),
    "quadrant_c_flagged_unsafe_frac": ("C", 20),
    "quadrant_d_flagged_unsafe_frac": ("D", 50),
}


def load_layer(stage, layer=FINAL_LAYER):
    with open(f"results/probes/{stage}_probe_results.json", encoding="utf-8") as f:
        results = json.load(f)
    matches = [r for r in results if r["layer"] == layer]
    if not matches:
        raise ValueError(f"Layer {layer} not found for {stage}")
    return matches[0]


def frac_to_ci(frac, n):
    if frac is None:
        return rate_with_ci(0, 0)
    return rate_with_ci(round(frac * n), n)


def main():
    print(f"Component 3 summary at layer {FINAL_LAYER} (final layer, all stages)")
    print("Fraction of each held-out quadrant the A-vs-B probe flags 'unsafe', Wilson 95% CI\n")
    for key, (label, n) in QUADRANTS.items():
        print(f"--- Quadrant {label} (n={n}) ---")
        for stage in STAGES:
            row = load_layer(stage)
            ci = frac_to_ci(row[key], n)
            print(f"  {stage}: {ci['rate']:.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
        print()


if __name__ == "__main__":
    main()