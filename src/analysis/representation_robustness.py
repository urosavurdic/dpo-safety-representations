"""`_pooled` vs `_final` representation sensitivity (WP-ReprRobust).

The frozen plan fixes the canonical direction on the ``_final`` (last prompt
token) activation. ``_pooled`` (mean of the last 5 tokens) is a reasonable
alternative pooling. This module recomputes the headline geometry on BOTH and
reports how much each conclusion moves, so a reader can see the pooling choice
is not load-bearing (or, if it is, that this is disclosed).

Reported per stage/layer:
  * ``cos(d_AD^final, d_AD^pooled)`` - do the two poolings give the same axis?
  * the A-D projection gap under each pooling and their ratio;
  * ``z_C`` under each pooling.

CPU-only. Exploratory (analysis_plan.md §2 "Exploratory": `_pooled`-token
sensitivity).
"""
from __future__ import annotations

import numpy as np

from src.analysis.control_directions import ad_direction, cosine_per_layer
from src.analysis.projection_trajectory import normalised_position, projection_by_quadrant


def compare_poolings(final_arr, pooled_arr, quadrants, splits):
    """final_arr / pooled_arr: (n_prompts, n_layers, hidden) for the SAME rows."""
    d_final = ad_direction(final_arr, quadrants, splits=splits)
    d_pooled = ad_direction(pooled_arr, quadrants, splits=splits)
    cos = cosine_per_layer(d_final, d_pooled)

    p_final = projection_by_quadrant(final_arr, quadrants, d_final)
    p_pooled = projection_by_quadrant(pooled_arr, quadrants, d_pooled)

    out = {
        "reference": "analysis_plan.md §2 (exploratory: _pooled sensitivity)",
        "cos_dAD_final_vs_pooled_per_layer": cos.tolist(),
        "cos_dAD_final_vs_pooled_mean_excl_layer0": float(np.mean(cos[1:])) if cos.size > 1 else float(cos.mean()),
        "per_layer": [],
    }
    n_layers = d_final.shape[0]
    for l in range(n_layers):
        entry = {"layer": l}
        for tag, p in (("final", p_final), ("pooled", p_pooled)):
            if "A" in p and "D" in p:
                gap = float(p["A"][l] - p["D"][l])
                entry[f"ad_gap_{tag}"] = gap
                if "C" in p:
                    z = normalised_position(
                        np.array([p["C"][l]]), np.array([p["A"][l]]), np.array([p["D"][l]])
                    )[0]
                    entry[f"z_C_{tag}"] = None if np.isnan(z) else float(z)
        if "ad_gap_final" in entry and entry.get("ad_gap_pooled"):
            entry["ad_gap_ratio_pooled_over_final"] = (
                entry["ad_gap_pooled"] / entry["ad_gap_final"]
                if entry["ad_gap_final"] else None
            )
        out["per_layer"].append(entry)
    return out


def summarize(comparison, *, cos_threshold=0.95):
    cos_mean = comparison["cos_dAD_final_vs_pooled_mean_excl_layer0"]
    return {
        "cos_mean_excl_layer0": cos_mean,
        "poolings_agree": cos_mean >= cos_threshold,
        "verdict": (
            "pooling choice is not load-bearing (final/pooled directions align)"
            if cos_mean >= cos_threshold else
            "pooling choice moves the direction - disclose as a limitation"
        ),
    }


STAGES = ["M0", "M1", "M2", "M3", "M3_direct", "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt"]


def _load_stage_pair(stage, act_dir):
    import json

    final = np.load(act_dir / f"{stage}_final.npy")
    pooled = np.load(act_dir / f"{stage}_pooled.npy")
    meta = json.loads((act_dir / f"{stage}_metadata.json").read_text(encoding="utf-8"))
    quadrants = np.array([r["quadrant"] for r in meta])
    splits = np.array([r.get("split") or "" for r in meta])
    return final, pooled, quadrants, splits


def main():  # pragma: no cover - CLI over on-disk activations; logic tested via compare_poolings/summarize
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--stages", nargs="+", default=STAGES)
    parser.add_argument("--out", default="results/summaries/representation_robustness.json")
    args = parser.parse_args()

    act_dir = Path(args.act_dir)
    out = {"reference": "analysis_plan.md §2 (exploratory: _pooled sensitivity)", "per_stage": {}}
    for stage in args.stages:
        if not (act_dir / f"{stage}_pooled.npy").exists() or not (act_dir / f"{stage}_final.npy").exists():
            print(f"  {stage}: skipped (need both _final.npy and _pooled.npy)")
            continue
        final, pooled, quadrants, splits = _load_stage_pair(stage, act_dir)
        cmp = compare_poolings(final, pooled, quadrants, splits)
        out["per_stage"][stage] = {**summarize(cmp), "detail": cmp}
        print(f"  {stage}: cos(final,pooled) mean = "
              f"{cmp['cos_dAD_final_vs_pooled_mean_excl_layer0']:.4f} -> "
              f"{out['per_stage'][stage]['verdict']}")

    agree = [v["poolings_agree"] for v in out["per_stage"].values()]
    out["all_stages_agree"] = bool(agree) and all(agree)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}  (all stages agree: {out['all_stages_agree']})")


if __name__ == "__main__":  # pragma: no cover
    main()
