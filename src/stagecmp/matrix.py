"""Assemble the stage-comparison artifact and its provenance sidecar.

Writes ``results/stagecmp/stage_comparison_{pooling}.json`` plus a
``*_binding.json`` recording the sha256 of every input it read, so a stale
activation file or a swapped judge file cannot go unnoticed.

Two things are deliberately recorded in ``diagnostics`` rather than left
implicit:

* ``scorer_coverage`` -- per stage, per scorer, how many behaviour rows are
  actually populated. Omitting this is how the project previously came to
  believe all nine stages carried StrongREJECT scores when only two did.
* ``row_order_verified`` -- the paired analyses are invalid without it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.common.activations import ACT_DIR, DIRECTIONS_DIR
from src.common.stages import ALL_STAGES
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.attribution import profile as depth_profile
from src.stagecmp.behav_metrics import (
    mcnemar,
    paired_difference,
    stage_profile,
    tv_and_agreement,
)
from src.stagecmp.contrasts import factorial_2x2, selectivity_2x2
from src.stagecmp.loaders import JUDGE_DEFAULT, assert_row_order, scorer_coverage
from src.stagecmp.pairs import (
    CONTRAST_CAVEAT,
    HEADLINE_PAIRS,
    TRANSITIONS_2X2,
    pair_key,
    unordered_pairs,
)
from src.v2_io import sha256_file, write_json_lf

OUT_DIR = Path("results/stagecmp")
QUADRANTS = ("A", "B", "C", "D")


def build(
    pooling: str = "final_token",
    metric: str = "withhold",
    act_dir=ACT_DIR,
    directions_dir=DIRECTIONS_DIR,
    judge_path=JUDGE_DEFAULT,
    stages=None,
    full_matrix: bool = True,
) -> dict:
    stages = list(stages or ALL_STAGES)
    record_ids = assert_row_order(stages, act_dir)

    behavioural = {s: stage_profile(s, judge_path) for s in stages}

    pairwise = {}
    pairs = unordered_pairs(stages) if full_matrix else list(HEADLINE_PAIRS)
    headline = set(HEADLINE_PAIRS) | {(b, a) for a, b in HEADLINE_PAIRS}
    for pre, post in pairs:
        entry = {}
        for q in QUADRANTS:
            cell = {
                "tv_and_agreement": tv_and_agreement(pre, post, q, judge_path),
                "mcnemar": mcnemar(pre, post, metric, q, judge_path),
            }
            # bootstrap CIs only for the eight pre-named headline pairs; 36
            # pairs x 4 quadrants of intervals would invite misreading
            if (pre, post) in headline:
                cell["paired_difference"] = paired_difference(
                    pre, post, metric, q, judge_path
                )
            entry[q] = cell
        pairwise[pair_key(pre, post)] = entry

    contrasts = {
        "by_quadrant": {
            q: factorial_2x2(metric, q, judge_path) for q in QUADRANTS
        },
        "selectivity": {
            "A_minus_D": selectivity_2x2(metric, "A", "D", judge_path),
            "C_minus_B": selectivity_2x2(metric, "C", "B", judge_path),
        },
    }

    depth = {
        s: depth_profile(s, act_dir, directions_dir, pooling) for s in stages
    }

    return {
        "status": STATUS_EXPLORATORY,
        "produced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pooling": pooling,
        "behavioural_metric": metric,
        "stages": stages,
        "caveat": CONTRAST_CAVEAT,
        "transitions_2x2": {k: list(v) for k, v in TRANSITIONS_2X2.items()},
        "behavioural": behavioural,
        "pairwise": pairwise,
        "contrasts_2x2": contrasts,
        "depth_attribution": depth,
        "diagnostics": {
            "row_order_verified": True,
            "n_records": len(record_ids),
            "scorer_coverage": scorer_coverage(stages, judge_path),
            "behavioural_half_is_pooling_independent": (
                "The behavioural tables read response TEXT, so they do not depend "
                "on the activation pooling. Only depth_attribution does."
            ),
            "ci_policy": (
                "Bootstrap CIs on paired differences are computed for the eight "
                "pre-named headline pairs only. All other cells carry point "
                "estimates, McNemar and TV."
            ),
        },
    }


def write(payload: dict, out_dir=OUT_DIR, act_dir=ACT_DIR, judge_path=JUDGE_DEFAULT) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooling = payload["pooling"]
    target = out_dir / f"stage_comparison_{pooling}.json"
    write_json_lf(target, payload)

    suffix = {"final_token": "final", "mean_last5": "pooled"}[pooling]
    inputs = {"judge_file": str(judge_path)}
    hashes = {"judge_file": sha256_file(judge_path)}
    for stage in payload["stages"]:
        p = Path(act_dir) / f"{stage}_{suffix}.npy"
        if p.exists():
            inputs[f"activations::{stage}"] = str(p)
            hashes[f"activations::{stage}"] = sha256_file(p)
    write_json_lf(
        out_dir / f"stage_comparison_{pooling}_binding.json",
        {
            "status": STATUS_EXPLORATORY,
            "produced_utc": payload["produced_utc"],
            "pooling": pooling,
            "inputs": inputs,
            "sha256": hashes,
        },
    )
    return target
