"""
Direction stability and drift analysis.

Tracks how the refusal direction changes (or stays stable) across training
stages, using Component 4's already-computed cosine_similarity.json.
"""
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from src.io_utils import load_json, write_json


def analyze_direction_stability(
    cosine_sim_path: str = "results/refusal_direction/cosine_similarity.json",
    output_path: str = "results/interpretability/direction_stability/stability_report.json",
) -> Dict[str, Any]:
    """
    cosine_sim_path's real schema (from src/analysis/eval_refusal_direction.py):
        {"vs_M0": {"M0": [n_layers floats], "M1": [...], "M2": [...], "M3": [...]},
         "adjacent": {"M0_vs_M1": [...], "M1_vs_M2": [...], "M2_vs_M3": [...]}}

    vs_M0/adjacent only contain whichever stages actually had activations
    extracted when eval_refusal_direction.py last ran -- a partial run (e.g.
    only M0's activations exist) writes only {"M0": [...]} , no M1/M2/M3 keys
    at all. This used to hardcode ["M0","M1","M2","M3"] and crash with
    KeyError on the first missing stage; now it only compares whichever of
    the canonical M0->M1->M2->M3 chain is actually present, and writes a
    reduced report (noting what's missing) rather than crashing the whole
    `src.reproduce direction` pipeline over one partial upstream run.
    """
    cosine_data = load_json(cosine_sim_path)

    results = {
        "metadata": {
            "analysis": "direction_stability_drift",
            "source_cosine_sim": cosine_sim_path,
        },
        "per_layer_stability": {},
        "stability_summary": {},
        "drift_dynamics": {},
    }

    CANONICAL_STAGE_ORDER = ["M0", "M1", "M2", "M3"]
    CANONICAL_ADJACENT_PAIRS = ["M0_vs_M1", "M1_vs_M2", "M2_vs_M3"]

    vs_m0 = cosine_data["vs_M0"]
    present_stages = [s for s in CANONICAL_STAGE_ORDER if s in vs_m0]
    missing_stages = [s for s in CANONICAL_STAGE_ORDER if s not in vs_m0]
    if missing_stages:
        results["metadata"]["missing_stages"] = missing_stages
        print(f"direction_stability: {missing_stages} not present in {cosine_sim_path} "
              "(their activations/direction haven't been (re)built yet) -- "
              f"reporting on {present_stages} only, not crashing.")

    if len(present_stages) < 2:
        results["metadata"]["note"] = (
            f"Only {present_stages} present -- need at least 2 stages (M0 plus one more) "
            "to compute any stability/drift comparison. Nothing to report yet."
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_json(results, output_path)
        print(f"Direction stability analysis (partial, no comparisons possible) saved to {output_path}")
        return results

    n_layers = len(vs_m0["M0"])
    for layer in range(n_layers):
        results["per_layer_stability"][layer] = {
            stage: float(vs_m0[stage][layer]) for stage in present_stages
        }

    final_stage = present_stages[-1]  # e.g. "M3" when the full chain is present, "M1" if that's all there is
    final_sims = [v[final_stage] for layer, v in results["per_layer_stability"].items() if layer != 0]
    results["stability_summary"] = {
        f"mean_similarity_M0_vs_{final_stage}": float(np.mean(final_sims)),
        "min_similarity": float(np.min(final_sims)),
        "max_similarity": float(np.max(final_sims)),
        "layers_high_stability": [l for l, v in results["per_layer_stability"].items() if v[final_stage] > 0.7],
        "layers_low_stability": [l for l, v in results["per_layer_stability"].items() if v[final_stage] < 0.5],
    }

    adj = cosine_data.get("adjacent", {})
    present_pairs = [p for p in CANONICAL_ADJACENT_PAIRS if p in adj]
    drift_by_layer = {}
    if present_pairs:
        for layer in range(n_layers):
            drift_by_layer[layer] = {
                pair: float(1.0 - adj[pair][layer]) for pair in present_pairs
            }
        results["drift_dynamics"]["per_layer_drift"] = drift_by_layer
        results["drift_dynamics"]["aggregate"] = {
            f"mean_drift_{pair}": float(np.mean([v[pair] for v in drift_by_layer.values()]))
            for pair in present_pairs
        }

        # Interpretation text generated FROM the numbers above, not hardcoded.
        agg = results["drift_dynamics"]["aggregate"]
        biggest_jump = max(agg, key=agg.get)
        results["interpretation"] = {
            "summary": (
                f"Largest average rotation occurs at {biggest_jump.replace('mean_drift_', '')} "
                f"(mean drift {agg[biggest_jump]:.3f}); the other adjacent-stage transitions are "
                f"smaller. Mean M0-vs-{final_stage} similarity across layers is "
                f"{results['stability_summary'][f'mean_similarity_M0_vs_{final_stage}']:.3f}, ranging "
                f"{results['stability_summary']['min_similarity']:.3f}-"
                f"{results['stability_summary']['max_similarity']:.3f} across layers."
            )
        }
    else:
        results["metadata"]["note"] = (
            results["metadata"].get("note", "") +
            " No adjacent-pair cosine data present -- drift_dynamics/interpretation skipped."
        ).strip()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(results, output_path)
    print(f"Direction stability analysis saved to {output_path}")
    return results


if __name__ == "__main__":
    analyze_direction_stability()