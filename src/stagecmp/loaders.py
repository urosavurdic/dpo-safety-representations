"""Binding-aware, coverage-aware loading for stage-vs-stage comparison.

Two invariants this module exists to enforce, both learned the hard way:

1. **Row order is identical across all nine stages.** ``results/DATA_STATE.md``
   §1 verifies it by ``record_id``. Every paired analysis in this package
   depends on it, and a mismatch would be silently wrong rather than noisy, so
   it is asserted rather than assumed.

2. **Judge coverage is per-scorer and very uneven.** The authoritative judge
   file was produced under ``scope="confirmatory"``, so only quadrant-C rows of
   M2 and M3 carry StrongREJECT and WildGuard; the other seven stages carry the
   regex classifier alone. A loader that reports an unscored stage as "rate 0"
   instead of "not covered" would manufacture a result. ``scorer_coverage``
   makes the gap explicit and ``require_scorer`` refuses to proceed without it.

``act_dir`` is always passed explicitly. A default that silently reaches for the
real 2 GB activation set makes tests depend on it (CONTRIBUTING.md).
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from src.common.activations import (
    POOLING_SUFFIX,
    load_activations,
    load_direction,
    load_metadata,
)
from src.common.stages import ALL_STAGES

JUDGE_DEFAULT = Path(
    "results/behavioral_judges/behavioral_judges_v2_20260907T043919Z.json"
)
BEHAVIOR_SUFFIX = "_behavior"
SCORERS = ("regex", "strong_reject", "wildguard")


class CoverageError(RuntimeError):
    """Raised when a requested scorer is not populated for a stage."""


# --------------------------------------------------------------- activations --

def load_stage_arrays(stage: str, pooling: str, act_dir) -> tuple[np.ndarray, list]:
    """``(activations, metadata)`` for one stage. Shape ``(n, n_layers, hidden)``."""
    if pooling not in POOLING_SUFFIX:
        raise ValueError(f"pooling must be one of {sorted(POOLING_SUFFIX)}, got {pooling!r}")
    arr = load_activations(stage, pooling, act_dir)
    meta = load_metadata(stage, act_dir)
    if arr.shape[0] != len(meta):
        raise ValueError(
            f"{stage}: activation rows {arr.shape[0]} != metadata rows {len(meta)}"
        )
    return arr, meta


def assert_row_order(stages: Iterable[str], act_dir) -> list[str]:
    """Every stage must expose the same ``record_id`` sequence. Returns it."""
    stages = list(stages)
    reference = None
    for stage in stages:
        ids = [row["record_id"] for row in load_metadata(stage, act_dir)]
        if reference is None:
            reference = ids
            first = stage
        elif ids != reference:
            mismatches = sum(1 for a, b in zip(ids, reference) if a != b)
            raise ValueError(
                f"row order differs between {first} and {stage} "
                f"({mismatches} positions, lengths {len(reference)} vs {len(ids)}). "
                "Paired analysis is invalid until this is fixed."
            )
    return reference


def quadrants_of(stage: str, act_dir) -> np.ndarray:
    return np.array([row["quadrant"] for row in load_metadata(stage, act_dir)])


def splits_of(stage: str, act_dir) -> np.ndarray:
    return np.array([row.get("split") or "" for row in load_metadata(stage, act_dir)])


def stage_direction(stage: str, directions_dir, layer=None) -> np.ndarray:
    """Unit A-D direction for a stage; ``(n_layers, hidden)`` or one layer."""
    return load_direction(stage, layer=layer, directions_dir=directions_dir)


# --------------------------------------------------------------------- judges --

@lru_cache(maxsize=4)
def _load_judge_file(path_str: str) -> tuple:
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])
    meta = {k: v for k, v in payload.items() if k != "records"} if isinstance(payload, dict) else {}
    return tuple(records), json.dumps(meta, sort_keys=True)


def judge_records(judge_path=JUDGE_DEFAULT) -> list[dict]:
    records, _meta = _load_judge_file(str(judge_path))
    return list(records)


def judge_meta(judge_path=JUDGE_DEFAULT) -> dict:
    _records, meta = _load_judge_file(str(judge_path))
    return json.loads(meta)


def behavior_rows(stage: str, judge_path=JUDGE_DEFAULT) -> dict[str, dict]:
    """``record_id -> judged row`` for one stage's un-intervened generations."""
    condition = f"{stage}{BEHAVIOR_SUFFIX}"
    out = {}
    for row in judge_records(judge_path):
        if row.get("condition") == condition:
            out[row["record_id"]] = row
    return out


def _scored(row: dict, scorer: str) -> bool:
    payload = row.get(scorer) or {}
    if not isinstance(payload, dict):
        return False
    if scorer == "regex":
        return payload.get("refused") is not None
    if scorer == "strong_reject":
        return payload.get("score") is not None
    if scorer == "wildguard":
        return payload.get("response_harm") is not None
    raise ValueError(f"unknown scorer {scorer!r}")


def scorer_coverage(stages=None, judge_path=JUDGE_DEFAULT) -> dict:
    """Per stage, per scorer: how many behaviour rows are actually populated.

    This is the check whose absence caused a wrong claim earlier in the project:
    654 rows exist for every stage, but only 104 of them carry StrongREJECT and
    WildGuard, and only on M2 and M3.
    """
    stages = list(stages or ALL_STAGES)
    by_stage = {}
    for stage in stages:
        rows = behavior_rows(stage, judge_path)
        counts = {"n_rows": len(rows)}
        per_quadrant = defaultdict(lambda: defaultdict(int))
        for scorer in SCORERS:
            counts[scorer] = sum(1 for r in rows.values() if _scored(r, scorer))
        for r in rows.values():
            for scorer in SCORERS:
                if _scored(r, scorer):
                    per_quadrant[scorer][r.get("quadrant", "?")] += 1
        counts["by_quadrant"] = {s: dict(q) for s, q in per_quadrant.items()}
        by_stage[stage] = counts
    return by_stage


def require_scorer(scorer: str, stages, judge_path=JUDGE_DEFAULT) -> None:
    """Refuse to run a scorer-based analysis on stages that lack that scorer."""
    coverage = scorer_coverage(stages, judge_path)
    missing = {
        stage: c[scorer]
        for stage, c in coverage.items()
        if c[scorer] < c["n_rows"]
    }
    if missing:
        detail = ", ".join(
            f"{s}: {n}/{coverage[s]['n_rows']}" for s, n in sorted(missing.items())
        )
        raise CoverageError(
            f"scorer {scorer!r} is not fully populated for {len(missing)} stage(s): "
            f"{detail}. Run the judging pass with --scope all before using it, or "
            f"use scorer='regex', which is complete for all nine stages."
        )
