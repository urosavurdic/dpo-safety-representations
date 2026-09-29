"""Behavioural distance between checkpoints on the identical 654 prompts.

Scorer-parametric from the start. The frozen regex classifier is complete for
all nine stages, so the full matrix is available with no GPU. StrongREJECT and
WildGuard currently cover only quadrant-C rows of M2 and M3 (see
``loaders.scorer_coverage``); once a judging pass fills them in, the same
functions produce the same tables under those scorers with no rewrite.

Two disciplines carried over from ``src/crossbranch/analyze.py``, both of which
exist because the obvious shortcut is wrong:

* total variation is never reported without the four-way distribution beside
  it, because a rise in the degenerate rate moves TV exactly as much as a rise
  in refusal does;
* TV is never reported without per-prompt agreement beside it, because two
  conditions can have identical label DISTRIBUTIONS while disagreeing on every
  individual prompt.
"""
from __future__ import annotations

import numpy as np

from src.analysis.mcnemar_direction_specificity import mcnemar_exact
from src.common.quadrants import CATEGORIES, QUADRANTS
from src.common.stats import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    paired_bootstrap_ci,
    rate_with_ci,
)
from src.crossbranch.analyze import distribution, total_variation
from src.stagecmp.loaders import JUDGE_DEFAULT, behavior_rows

#: per-prompt scalars this module can extract, and which scorer each needs
METRICS = {
    "withhold": "regex",        # refusal OR soft deflection
    "refusal": "regex",         # refusal only
    "degenerate": "regex",
    "sr_score": "strong_reject",
    "wg_harm": "wildguard",
    "wg_refusal": "wildguard",
}
BINARY_METRICS = {"withhold", "refusal", "degenerate", "wg_harm", "wg_refusal"}


def label_of(row: dict) -> str | None:
    """Four-way regex label, in the frozen precedence order from
    ``common/refusal_classifier.classify_completion``:
    degenerate > refusal > soft_deflection > comply."""
    rx = row.get("regex") or {}
    if rx.get("refused") is None:
        return None
    if rx.get("degenerate"):
        return "degenerate"
    if rx.get("refused"):
        return "refusal"
    if rx.get("soft_deflection"):
        return "soft_deflection"
    return "comply"


def value_of(row: dict, metric: str) -> float | None:
    """One per-prompt scalar, or ``None`` when that scorer did not run."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; known: {sorted(METRICS)}")
    if METRICS[metric] == "regex":
        label = label_of(row)
        if label is None:
            return None
        if metric == "withhold":
            return float(label in ("refusal", "soft_deflection"))
        if metric == "refusal":
            return float(label == "refusal")
        return float(label == "degenerate")
    if metric == "sr_score":
        return (row.get("strong_reject") or {}).get("score")
    wg = row.get("wildguard") or {}
    return wg.get("response_harm" if metric == "wg_harm" else "response_refusal")


def per_prompt(stage: str, metric: str, judge_path=JUDGE_DEFAULT) -> dict[str, float]:
    """``record_id -> value``, omitting prompts this scorer did not cover."""
    out = {}
    for rid, row in behavior_rows(stage, judge_path).items():
        v = value_of(row, metric)
        if v is not None:
            out[rid] = float(v)
    return out


def quadrant_of(stage: str, judge_path=JUDGE_DEFAULT) -> dict[str, str]:
    return {rid: row["quadrant"] for rid, row in behavior_rows(stage, judge_path).items()}


# ------------------------------------------------------------ per stage ----- #

def stage_profile(stage: str, judge_path=JUDGE_DEFAULT) -> dict:
    """Four-way label distribution and rates per quadrant for one stage."""
    rows = behavior_rows(stage, judge_path)
    by_quadrant = {q: [] for q in QUADRANTS}
    for row in rows.values():
        label = label_of(row)
        if label is not None:
            by_quadrant[row["quadrant"]].append(label)

    out = {"stage": stage, "n_rows": len(rows), "per_quadrant": {}}
    for q, labels in by_quadrant.items():
        n = len(labels)
        counts = {c: labels.count(c) for c in CATEGORIES}
        withhold = counts["refusal"] + counts["soft_deflection"]
        out["per_quadrant"][q] = {
            "n": n,
            "four_way_counts": counts,
            "four_way": {c: (counts[c] / n if n else None) for c in CATEGORIES},
            "withhold": rate_with_ci(withhold, n),
            "refusal": rate_with_ci(counts["refusal"], n),
            "degenerate": rate_with_ci(counts["degenerate"], n),
        }
    return out


# --------------------------------------------------------------- pairwise ---- #

def _shared(stage_x, stage_y, metric, judge_path, quadrant=None):
    """Complete units only: a prompt unusable at EITHER stage is dropped whole,
    the same rule CF1 uses."""
    vx = per_prompt(stage_x, metric, judge_path)
    vy = per_prompt(stage_y, metric, judge_path)
    quad = quadrant_of(stage_x, judge_path)
    ids = sorted(set(vx) & set(vy))
    if quadrant is not None:
        ids = [i for i in ids if quad.get(i) == quadrant]
    return ids, np.array([vx[i] for i in ids]), np.array([vy[i] for i in ids])


def paired_difference(
    stage_x: str,
    stage_y: str,
    metric: str = "withhold",
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap CI on ``mean(y - x)`` over complete prompt units."""
    ids, x, y = _shared(stage_x, stage_y, metric, judge_path, quadrant)
    n_x = len(per_prompt(stage_x, metric, judge_path))
    n_y = len(per_prompt(stage_y, metric, judge_path))
    result = paired_bootstrap_ci(y - x, b=b, seed=seed)
    result.update(
        {
            "pre": stage_x,
            "post": stage_y,
            "metric": metric,
            "quadrant": quadrant or "all",
            "mean_pre": float(x.mean()) if len(x) else None,
            "mean_post": float(y.mean()) if len(y) else None,
            "dropped_incomplete": {"pre_rows": n_x, "post_rows": n_y, "shared": len(ids)},
        }
    )
    return result


def mcnemar(
    stage_x: str,
    stage_y: str,
    metric: str = "withhold",
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
) -> dict:
    """Exact-binomial McNemar on a paired binary outcome, keyed on record_id.

    ``b`` and ``c`` are DISCORDANT-PAIR COUNTS here. In the bootstrap artifacts
    of this repo the key ``b`` means the bootstrap replicate count instead --
    see results/README.md. Same letter, unrelated meanings.
    """
    if metric not in BINARY_METRICS:
        raise ValueError(f"McNemar needs a binary metric; {metric!r} is not one")
    ids, x, y = _shared(stage_x, stage_y, metric, judge_path, quadrant)
    xb, yb = x.astype(bool), y.astype(bool)
    b = int(np.sum(xb & ~yb))   # yes at x, no at y
    c = int(np.sum(~xb & yb))   # no at x, yes at y
    return {
        "pre": stage_x,
        "post": stage_y,
        "metric": metric,
        "quadrant": quadrant or "all",
        "n_pairs": len(ids),
        "b_def": f"{stage_x}=1 and {stage_y}=0 (DISCORDANT COUNT, not bootstrap B)",
        "c_def": f"{stage_x}=0 and {stage_y}=1",
        "b": b,
        "c": c,
        "n_discordant": b + c,
        "p_exact": mcnemar_exact(b, c),
    }


def tv_and_agreement(
    stage_x: str,
    stage_y: str,
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
) -> dict:
    """Four-way TV distance, the two distributions it came from, and per-prompt
    agreement. All three together, never TV alone."""
    rows_x = behavior_rows(stage_x, judge_path)
    rows_y = behavior_rows(stage_y, judge_path)
    ids = sorted(set(rows_x) & set(rows_y))
    if quadrant is not None:
        ids = [i for i in ids if rows_x[i]["quadrant"] == quadrant]
    lx = [label_of(rows_x[i]) for i in ids]
    ly = [label_of(rows_y[i]) for i in ids]
    keep = [k for k, (a, b_) in enumerate(zip(lx, ly)) if a and b_]
    lx = [lx[k] for k in keep]
    ly = [ly[k] for k in keep]
    if not lx:
        return {"pre": stage_x, "post": stage_y, "quadrant": quadrant or "all", "n": 0}
    p, q = distribution(lx), distribution(ly)
    agree = float(np.mean([a == b_ for a, b_ in zip(lx, ly)]))
    return {
        "pre": stage_x,
        "post": stage_y,
        "quadrant": quadrant or "all",
        "n": len(lx),
        "total_variation": total_variation(p, q),
        "four_way_pre": dict(zip(CATEGORIES, [float(v) for v in p])),
        "four_way_post": dict(zip(CATEGORIES, [float(v) for v in q])),
        "per_prompt_agreement": agree,
        "reading": (
            "TV is an aggregate distance: it moves when the degenerate rate "
            "moves, and it can be near zero while per-prompt agreement is at "
            "chance. Read all three fields together."
        ),
    }
