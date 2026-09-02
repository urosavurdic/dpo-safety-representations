"""Projection-magnitude trajectory statistic (WP-Geom), analysis_plan.md §4.5.

Per quadrant q, stage s, layer l:
    p_{q,s,l} = (1/n_q) sum_{i in q} h_i^{s,l} . d^{s,l}
with d^{s,l} the unit A-D direction oriented D->A (estimation split only).
Then  p_{A} - p_{D} = ||m_A - m_D||  and C's normalised position is
    z_{C,s,l} = (p_C - p_D) / (p_A - p_D)
z_C ~ 0 => C near D; z_C ~ 1 => C near A; increasing across stages => C moves
toward A along the A-D contrast. z is reported MISSING when the denominator is
zero or numerically negligible (< 1e-6 * ||m_A - m_D||... here relative 1e-6).

Also computes fixed-reference trajectories: an M1-reference axis and an
M3-reference axis, each computed once and applied to every stage.
"""
from __future__ import annotations

import numpy as np

from src.analysis.control_directions import ad_direction

DIRECTION_SPLIT = "direction_estimation"
NEGLIGIBLE_REL = 1e-6


def projection_by_quadrant(pooled, quadrants, direction):
    """direction: (n_layers, hidden). Returns {quadrant: (n_layers,) mean proj}."""
    out = {}
    for q in ("A", "B", "C", "D"):
        mask = quadrants == q
        if not mask.any():
            continue
        proj = np.einsum("nlh,lh->nl", pooled[mask], direction)
        out[q] = proj.mean(0)
    return out


def normalised_position(p_target, p_a, p_d):
    """z = (p_target - p_d) / (p_a - p_d), missing when the gap is negligible."""
    denom = p_a - p_d
    scale = np.maximum(np.abs(p_a), np.abs(p_d))
    z = np.where(
        np.abs(denom) <= NEGLIGIBLE_REL * np.where(scale == 0, 1.0, scale),
        np.nan,
        (p_target - p_d) / np.where(denom == 0, np.nan, denom),
    )
    return z


def stage_trajectory(pooled, quadrants, splits):
    """Stage-specific: estimate d on this stage, project every quadrant onto it."""
    d = ad_direction(pooled, quadrants, splits=splits)
    p = projection_by_quadrant(pooled, quadrants, d)
    result = {"projection_by_quadrant": {q: v.tolist() for q, v in p.items()}}
    if "A" in p and "D" in p:
        gap = p["A"] - p["D"]
        result["ad_gap_per_layer"] = gap.tolist()
        if "C" in p:
            result["z_C_per_layer"] = _nan_to_none(normalised_position(p["C"], p["A"], p["D"]))
        if "B" in p:
            result["z_B_per_layer"] = _nan_to_none(normalised_position(p["B"], p["A"], p["D"]))
    return result


def fixed_reference_trajectory(pooled, quadrants, reference_direction):
    """Project every quadrant of ``pooled`` onto a direction estimated at some
    OTHER (reference) stage."""
    p = projection_by_quadrant(pooled, quadrants, reference_direction)
    out = {"projection_by_quadrant": {q: v.tolist() for q, v in p.items()}}
    if "A" in p and "D" in p:
        out["ad_gap_per_layer"] = (p["A"] - p["D"]).tolist()
        if "C" in p:
            out["z_C_per_layer"] = _nan_to_none(normalised_position(p["C"], p["A"], p["D"]))
    return out


def _nan_to_none(arr):
    return [None if np.isnan(x) else float(x) for x in np.atleast_1d(arr)]


def build_trajectories(stage_activations: dict):
    """stage_activations: {stage: (pooled, quadrants, splits)}.
    Returns stage-specific + M1-reference + M3-reference trajectories."""
    result = {"stage_specific": {}, "fixed_reference": {}}
    directions = {}
    for stage, (pooled, quads, splits) in stage_activations.items():
        result["stage_specific"][stage] = stage_trajectory(pooled, quads, splits)
        directions[stage] = ad_direction(pooled, quads, splits=splits)

    for ref in ("M1", "M3"):
        if ref not in directions:
            continue
        result["fixed_reference"][f"{ref}_reference"] = {
            stage: fixed_reference_trajectory(pooled, quads, directions[ref])
            for stage, (pooled, quads, splits) in stage_activations.items()
        }
    return result


def bootstrap_z_c(
    pooled, quadrants, splits, layer, *, n_boot=10000, seed=20260904
):
    """Prompt-level percentile bootstrap CI on z_C at one layer. Resamples
    A_est / D_est / C jointly by prompt; the direction is re-estimated per
    replicate."""
    est = np.array([str(s) == DIRECTION_SPLIT for s in splits])
    a_idx = np.flatnonzero((quadrants == "A") & est)
    d_idx = np.flatnonzero((quadrants == "D") & est)
    c_idx = np.flatnonzero(quadrants == "C")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        a_s = rng.choice(a_idx, len(a_idx), replace=True)
        d_s = rng.choice(d_idx, len(d_idx), replace=True)
        c_s = rng.choice(c_idx, len(c_idx), replace=True)
        layer_h = pooled[:, layer]
        d_vec = layer_h[a_s].mean(0) - layer_h[d_s].mean(0)
        n = np.linalg.norm(d_vec)
        if n == 0:
            continue
        d_vec = d_vec / n
        p_a = float((layer_h[a_s] @ d_vec).mean())
        p_d = float((layer_h[d_s] @ d_vec).mean())
        p_c = float((layer_h[c_s] @ d_vec).mean())
        z = normalised_position(np.array([p_c]), np.array([p_a]), np.array([p_d]))[0]
        if not np.isnan(z):
            vals.append(float(z))
    vals = np.asarray(vals)
    return {
        "layer": int(layer), "n_boot": int(len(vals)), "seed": seed,
        "interval": "percentile",
        "mean": float(vals.mean()) if len(vals) else None,
        "ci_low": float(np.percentile(vals, 2.5)) if len(vals) else None,
        "ci_high": float(np.percentile(vals, 97.5)) if len(vals) else None,
    }


def main():  # pragma: no cover - [exec:T4], needs regenerated 654-row activations
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--stages", nargs="+",
                        default=["M0", "M1", "M2", "M3", "M3_direct",
                                 "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt"])
    parser.add_argument("--out", default="results/refusal_direction/projection_trajectory.json")
    args = parser.parse_args()

    act = Path(args.act_dir)
    stages = {}
    for s in args.stages:
        mp = act / f"{s}_metadata.json"
        if not mp.exists():
            continue
        arr = np.load(act / f"{s}_final.npy")
        meta = json.loads(mp.read_text(encoding="utf-8"))
        stages[s] = (arr, np.array([r["quadrant"] for r in meta]),
                     np.array([r.get("split") or "" for r in meta]))
    out = build_trajectories(stages)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out} for {sorted(stages)}")


if __name__ == "__main__":  # pragma: no cover
    main()
