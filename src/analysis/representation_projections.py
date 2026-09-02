"""Canonical `_final` direction + per-prompt & fixed-reference projections (WP-Repr).

analysis_plan.md §4 fixes the canonical direction on the RAW ``_final``
activation (last prompt token), estimated on the ``direction_estimation`` split:

    d^{s,l} = (m_A^{s,l} - m_D^{s,l}) / ||m_A^{s,l} - m_D^{s,l}||

This module persists, per stage:
  * the per-layer canonical ``_final`` direction (``{stage}_direction_final.npy``);
  * **per-prompt** projections of every eval row onto that stage's own
    direction (not just per-quadrant means) - needed for the §4.5 trajectory
    bootstrap and for matched C-pair deltas;
  * **fixed-reference** projections: every stage's rows projected onto the
    M1-reference axis and onto the M3-reference axis (a fixed axis vs a
    re-estimated one answer different questions and are reported side by side).

CPU-only. Reads ``results/activations/{stage}_final.npy`` +
``{stage}_metadata.json`` (falls back to ``_pooled`` only with a loud warning).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.control_directions import ad_direction
from src.analysis.projection_trajectory import build_trajectories

ACT_DIR = Path("results/activations")
OUT_DIR = Path("results/refusal_direction")
SEQUENTIAL_STAGES = ["M0", "M1", "M2", "M3"]
ALL_STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]


def load_stage_final(stage, act_dir=ACT_DIR):
    """Prefer the canonical ``_final`` position (analysis_plan.md §4)."""
    final_p = act_dir / f"{stage}_final.npy"
    pooled_p = act_dir / f"{stage}_pooled.npy"
    if final_p.exists():
        arr = np.load(final_p)
        position = "final"
    elif pooled_p.exists():
        print(f"  WARNING: {stage} has no _final.npy; falling back to _pooled "
              "(analysis_plan.md §4 fixes the canonical direction on _final).")
        arr = np.load(pooled_p)
        position = "pooled_fallback"
    else:
        raise FileNotFoundError(f"no activations for {stage} in {act_dir}")
    meta = json.loads((act_dir / f"{stage}_metadata.json").read_text(encoding="utf-8"))
    quads = np.array([r["quadrant"] for r in meta])
    splits = np.array([r.get("split") or "" for r in meta])
    ids = [r.get("record_id") for r in meta]
    return arr, quads, splits, ids, position


def per_prompt_projections(pooled, quadrants, direction):
    """(n_prompts, n_layers) projection of every row onto `direction`."""
    return np.einsum("nlh,lh->nl", pooled, direction)


def build_stage_projections(stage_activations):
    """stage_activations: {stage: (arr, quads, splits, ids)}.
    Returns a JSON-able dict with per-prompt + fixed-reference projections."""
    directions = {}
    per_prompt = {}
    for stage, (arr, quads, splits, ids) in stage_activations.items():
        d = ad_direction(arr, quads, splits=splits)
        directions[stage] = d
        proj = per_prompt_projections(arr, quads, d)
        per_prompt[stage] = [
            {"record_id": ids[i], "quadrant": quads[i].item() if hasattr(quads[i], "item") else quads[i],
             "split": splits[i].item() if hasattr(splits[i], "item") else splits[i],
             "projection_per_layer": proj[i].tolist()}
            for i in range(arr.shape[0])
        ]

    fixed_ref = {}
    for ref in ("M1", "M3"):
        if ref not in directions:
            continue
        fixed_ref[f"{ref}_reference"] = {}
        for stage, (arr, quads, splits, ids) in stage_activations.items():
            proj = per_prompt_projections(arr, quads, directions[ref])
            fixed_ref[f"{ref}_reference"][stage] = {
                q: float(proj[quads == q].mean()) if (quads == q).any() else None
                for q in ("A", "B", "C", "D")
            }

    traj_input = {s: (a, q, sp) for s, (a, q, sp, _i) in stage_activations.items()}
    return {
        "canonical_position": "final",
        "reference": "analysis_plan.md §4 / §4.5",
        "per_prompt_projections": per_prompt,
        "fixed_reference_quadrant_means": fixed_ref,
        "trajectories": build_trajectories(traj_input),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", default=str(ACT_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES)
    args = parser.parse_args()

    act_dir = Path(args.act_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_acts = {}
    for stage in args.stages:
        if not (act_dir / f"{stage}_metadata.json").exists():
            print(f"  {stage}: SKIPPED, no activations")
            continue
        arr, quads, splits, ids, position = load_stage_final(stage, act_dir)
        stage_acts[stage] = (arr, quads, splits, ids)
        d = ad_direction(arr, quads, splits=splits)
        np.save(out_dir / f"{stage}_direction_final.npy", d)
        print(f"  {stage}: direction_final {d.shape} (position={position})")

    if not stage_acts:
        print("no stages available; nothing written.")
        return

    payload = build_stage_projections(stage_acts)
    (out_dir / "per_prompt_projections.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'per_prompt_projections.json'} "
          f"({sum(len(v) for v in payload['per_prompt_projections'].values())} rows)")


if __name__ == "__main__":
    main()
