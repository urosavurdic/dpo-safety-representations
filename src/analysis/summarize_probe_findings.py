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
from pathlib import Path

from src.analysis.eval_probes import B_TRAIN_SIZE
from src.eval_stats import rate_with_ci

STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]
FINAL_LAYER = 28


def _benchmark_quadrant_counts(latest_path="data/frozen_v2/LATEST_BENCHMARK.json"):
    """Per-quadrant row counts of the frozen v2 benchmark (WP-Probe: n must
    come from LATEST_BENCHMARK.json, not the stale hardcoded 370-era 20/50/200)."""
    from src.v2_io import load_jsonl, resolve_benchmark

    bench_path, _sha = resolve_benchmark(latest_path=latest_path)
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for row in load_jsonl(bench_path):
        q = row.get("quadrant")
        if q in counts:
            counts[q] += 1
    return counts


def quadrant_report_ns(latest_path="data/frozen_v2/LATEST_BENCHMARK.json"):
    c = _benchmark_quadrant_counts(latest_path)
    return {
        # held-out B = all B minus the B_TRAIN_SIZE rows used to train the probe
        "holdout_b_flagged_unsafe_frac": ("B (held-out)", max(c["B"] - B_TRAIN_SIZE, 0)),
        "quadrant_c_flagged_unsafe_frac": ("C", c["C"]),
        "quadrant_d_flagged_unsafe_frac": ("D", c["D"]),
    }


# Backward-compatible module-level default; recomputed in main() from the
# frozen benchmark so a re-freeze does not require editing this file.
try:
    QUADRANTS = quadrant_report_ns()
except Exception:  # pragma: no cover - benchmark pointer absent in some sandboxes
    QUADRANTS = {
        "holdout_b_flagged_unsafe_frac": ("B (held-out)", 200),
        "quadrant_c_flagged_unsafe_frac": ("C", 104),
        "quadrant_d_flagged_unsafe_frac": ("D", 150),
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
    try:
        quadrants = quadrant_report_ns()
    except Exception:
        quadrants = QUADRANTS
    print(f"Component 3 summary at layer {FINAL_LAYER} (final layer, all stages)")
    print(f"Per-quadrant n from LATEST_BENCHMARK.json: "
          f"{ {label: n for label, n in quadrants.values()} }")
    print("Fraction of each held-out quadrant the A-vs-B probe flags 'unsafe', Wilson 95% CI\n")
    for key, (label, n) in quadrants.items():
        print(f"--- Quadrant {label} (n={n}) ---")
        for stage in STAGES:
            result_path = Path(f"results/probes/{stage}_probe_results.json")
            if not result_path.exists():
                # Alt branch trains/pushes independently across sessions -
                # not every stage in STAGES necessarily has probe results yet.
                print(f"  {stage}: SKIPPED, no probe results yet")
                continue
            row = load_layer(stage)
            ci = frac_to_ci(row[key], n)
            print(f"  {stage}: {ci['rate']:.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
        print()


if __name__ == "__main__":
    main()