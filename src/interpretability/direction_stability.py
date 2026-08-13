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

    vs_m0 = cosine_data["vs_M0"]
    n_layers = len(vs_m0["M0"])
    for layer in range(n_layers):
        results["per_layer_stability"][layer] = {
            stage: float(vs_m0[stage][layer]) for stage in ["M0", "M1", "M2", "M3"]
        }

    m3_sims = [v["M3"] for layer, v in results["per_layer_stability"].items() if layer != 0]
    results["stability_summary"] = {
        "mean_similarity_M0_vs_M3": float(np.mean(m3_sims)),
        "min_similarity": float(np.min(m3_sims)),
        "max_similarity": float(np.max(m3_sims)),
        "layers_high_stability": [l for l, v in results["per_layer_stability"].items() if v["M3"] > 0.7],
        "layers_low_stability": [l for l, v in results["per_layer_stability"].items() if v["M3"] < 0.5],
    }

    adj = cosine_data["adjacent"]
    drift_by_layer = {}
    for layer in range(n_layers):
        drift_by_layer[layer] = {
            pair: float(1.0 - adj[pair][layer]) for pair in ["M0_vs_M1", "M1_vs_M2", "M2_vs_M3"]
        }
    results["drift_dynamics"]["per_layer_drift"] = drift_by_layer
    results["drift_dynamics"]["aggregate"] = {
        f"mean_drift_{pair}": float(np.mean([v[pair] for v in drift_by_layer.values()]))
        for pair in ["M0_vs_M1", "M1_vs_M2", "M2_vs_M3"]
    }

    # Interpretation text generated FROM the numbers above, not hardcoded.
    agg = results["drift_dynamics"]["aggregate"]
    biggest_jump = max(agg, key=agg.get)
    results["interpretation"] = {
        "summary": (
            f"Largest average rotation occurs at {biggest_jump.replace('mean_drift_', '')} "
            f"(mean drift {agg[biggest_jump]:.3f}); the other adjacent-stage transitions are "
            f"smaller. Mean M0-vs-M3 similarity across layers is "
            f"{results['stability_summary']['mean_similarity_M0_vs_M3']:.3f}, ranging "
            f"{results['stability_summary']['min_similarity']:.3f}-"
            f"{results['stability_summary']['max_similarity']:.3f} across layers."
        )
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(results, output_path)
    print(f"Direction stability analysis saved to {output_path}")
    return results


if __name__ == "__main__":
    analyze_direction_stability()